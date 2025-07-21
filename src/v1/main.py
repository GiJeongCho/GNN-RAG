

# === RAG 기능: PDF 추출 · 임베딩 · 검색 ===
from pathlib import Path
import time
from pydantic import BaseModel
from typing import List

# 리소스 디렉터리 내 경로 정의
EXTRACTED_TEXT_DIR = Path(resource_dir) / "extracted_texts"
META_JSON_PATH = EXTRACTED_TEXT_DIR / "_extraction_meta.json"
MODEL_PATH = Path(resource_dir) / "embedding_bge_m3"

# ----------------------------
# 요청 모델
# ----------------------------
class PDFExtractRequest(BaseModel):
    dir_path: str  # PDF들이 위치한 루트 경로

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    user_level: int = 1

# ----------------------------
# 유틸리티: mean-pooling (HF 모델)
# ----------------------------
import torch, torch.nn.functional as F

def _mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts

# ----------------------------
# 1) PDF → 텍스트 추출
# ----------------------------
async def extract_pdfs(req: PDFExtractRequest):
    """주어진 디렉터리에서 PDF를 찾아 텍스트(.txt)와 메타 JSON을 생성"""
    import fitz, json  # 로컬 import로 가볍게
    from tqdm import tqdm

    root_dir = Path(req.dir_path)
    if not root_dir.exists():
        return {"error": f"경로가 존재하지 않습니다: {req.dir_path}"}

    # 출력 디렉터리 준비
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    # 이미 처리된 메타 로드
    done_files = {}
    if META_JSON_PATH.exists():
        done_files = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    new_meta = {}
    pdf_paths = list(root_dir.rglob("*.pdf"))
    if not pdf_paths:
        return {"message": "처리할 PDF가 없습니다."}

    for pdf_path in tqdm(pdf_paths, desc="PDF 전처리"):
        pdf_rel = pdf_path.relative_to(root_dir)
        txt_path = EXTRACTED_TEXT_DIR / pdf_rel.with_suffix(".txt")
        key = str(pdf_rel)
        if key in done_files and txt_path.exists():
            new_meta[key] = done_files[key]
            continue
        try:
            doc = fitz.open(pdf_path)
            text_pages = [p.get_text("text").strip() for p in doc]
            pdf_text = "\n\n".join(text_pages)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(pdf_text, encoding="utf-8")

            # 보안레벨 추출 (securityLevelN 폴더명 가정, 없으면 1)
            level_folder = pdf_rel.parts[0] if len(pdf_rel.parts) else "securityLevel1"
            try:
                security_level = int(level_folder.replace("securityLevel", ""))
            except ValueError:
                security_level = 1

            lines = pdf_text.splitlines()
            info = {
                "chars": len(pdf_text),
                "lines": len(lines),
                "preview": pdf_text[:200].replace("\n", " ") + "…",
                "security_level": security_level,
            }
            new_meta[key] = info
        except Exception as e:
            new_meta[key] = {"error": str(e)}

    META_JSON_PATH.write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"message": "PDF 추출 완료", "pdf_count": len(pdf_paths), "meta_path": str(META_JSON_PATH)}

# ----------------------------
# 2) 텍스트 임베딩 & Milvus 인제스트
# ----------------------------
async def ingest_embeddings():
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
    from transformers import AutoTokenizer, AutoModel
    import json, numpy as np

    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME = "localhost", "19530", "pdf_chunks"
    MAX_TOKENS, OVERLAP = 512, 64

    # 메타 확인
    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF 추출을 수행하세요."}
    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    # Milvus 연결 & 초기화
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # 모델 준비
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16).to(device).eval()

    # 임베딩 차원 계산
    dummy_inp = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        dummy_out = model(**dummy_inp)
    emb_dim = _mean_pooling(dummy_out, dummy_inp["attention_mask"]).shape[1]

    # Milvus 스키마
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb_dim),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_idx", dtype=DataType.INT64),
        FieldSchema(name="security_level", dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description="PDF 청크 + 보안레벨")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}})
    collection.load()

    # 청크 함수
    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    # 인제스트 루프
    pk_counter = 0
    for txt_path in EXTRACTED_TEXT_DIR.rglob("*.txt"):
        rel_txt = txt_path.relative_to(EXTRACTED_TEXT_DIR)
        rel_pdf = rel_txt.with_suffix(".pdf").as_posix()
        if rel_pdf not in extraction_meta:
            continue
        sec_level = extraction_meta[rel_pdf]["security_level"]
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            inputs = tokenizer(chunk, truncation=True, padding="longest", max_length=MAX_TOKENS, return_tensors="pt").to(device)
            with torch.no_grad():
                outs = model(**inputs)
            vec = _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
            collection.insert([[pk_counter], [vec.tolist()], [str(rel_txt)], [idx], [sec_level]])
            pk_counter += 1

    return {"message": "Ingest 완료", "inserted_chunks": pk_counter}

# ----------------------------
# 3) 검색
# ----------------------------
async def search_documents(req: RAGSearchRequest):
    from pymilvus import connections, Collection
    from transformers import AutoTokenizer, AutoModel
    import json, numpy as np

    start_time = time.perf_counter()

    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME = "localhost", "19530", "pdf_chunks"
    MAX_TOKENS, OVERLAP = 512, 64

    # 메타 로드
    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다."}
    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    # Milvus 연결
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16).to(device).eval()

    # 쿼리 임베딩
    inputs = tokenizer(req.query, truncation=True, padding="longest", max_length=MAX_TOKENS, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    q_emb = _mean_pooling(outputs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")

    # 검색
    expr = f"security_level <= {req.user_level}"
    results = collection.search(data=[q_emb.tolist()], anns_field="embedding", param={"metric_type": "IP", "params": {"ef": 100}}, limit=req.top_k, expr=expr, output_fields=["path", "chunk_idx", "security_level"])

    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP):
        words = text.split()
        chunks, start = [], 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    hits = []
    for hit in results[0]:
        path = hit.entity.get("path")
        cidx = hit.entity.get("chunk_idx")
        sec_level = hit.entity.get("security_level")
        full_txt = (EXTRACTED_TEXT_DIR / path).read_text(encoding="utf-8")
        snippet = chunk_text(full_txt)[cidx]
        hits.append({
            "score": hit.score,
            "path": path,
            "chunk_idx": cidx,
            "security_level": sec_level,
            "snippet": snippet,
        })

    context = "\n---\n".join([h['snippet'] for h in hits])
    prompt = f"사용자 질의: {req.query}\n\n관련 문서 스니펫:\n{context}\n\n위 내용을 참고하여 응답해 주세요."

    elapsed = round(time.perf_counter() - start_time, 4)
    return {"elapsed_sec": elapsed, "hits": hits, "prompt": prompt}

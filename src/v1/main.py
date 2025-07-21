from pathlib import Path
import time
import os
from pydantic import BaseModel
from typing import List

# === RAG 기능: PDF 추출 · 임베딩 · 검색 ===
from pathlib import Path
import time
import os
from pydantic import BaseModel
from typing import List

# 리소스 디렉터리(환경변수로 재정의 가능)
resource_dir = os.getenv('RESOURCE_DIR', 'src/v1/resources')

# 리소스 디렉터리 내 경로 정의
EXTRACTED_TEXT_DIR = Path(resource_dir) / "extracted_texts"
META_JSON_PATH = EXTRACTED_TEXT_DIR / "_extraction_meta.json"
MODEL_PATH = Path(resource_dir) / "model" / "embedding_bge_m3"

# ----------------------------
# 요청 모델
# ----------------------------
class PDFExtractRequest(BaseModel):
    dir_path: str  # PDF들이 위치한 루트 경로

class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    user_level: int = 1

# 단일 PDF만 임베딩 요청
class SinglePDFIngestRequest(BaseModel):
    pdf_path: str  # 새 PDF의 절대 또는 상대 경로 ( .pdf )

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
# 공통 헬퍼 – 파일명에서 doc_id / version 추출
# ----------------------------

def _parse_doc_version(stem: str):
    """파일 스템에서 마지막 '_' 토큰이 4or8자리 숫자면 version으로 간주"""
    if "_" in stem:
        base, cand = stem.rsplit("_", 1)
        if cand.isdigit() and len(cand) in (4, 8):
            return base, int(cand)
    return stem, 0

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

            # ----- 문서 ID & 버전 파싱 -----
            doc_id_part, version_num = _parse_doc_version(pdf_rel.stem)

            lines = pdf_text.splitlines()
            info = {
                "chars": len(pdf_text),
                "lines": len(lines),
                "preview": pdf_text[:200].replace("\n", " ") + "…",
                "security_level": security_level,
                "doc_id": doc_id_part,
                "version": version_num,
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

    collection_exists = utility.has_collection(COLLECTION_NAME)

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
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="version", dtype=DataType.INT64),
    ]
    if not collection_exists:
        schema = CollectionSchema(fields, description="PDF 청크 + 보안레벨 + 버전")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        collection.create_index(field_name="embedding", index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}})
    else:
        collection = Collection(COLLECTION_NAME)
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
        meta_entry = extraction_meta[rel_pdf]
        sec_level = meta_entry["security_level"]
        doc_id = meta_entry.get("doc_id")
        version = meta_entry.get("version", 0)

        # ── 레거시 메타 보정 (doc_id 또는 version 누락) ──
        if doc_id is None or version == 0:
            _id_part, _ver_num = _parse_doc_version(rel_txt.stem)

            if doc_id is None:
                doc_id = _id_part
                meta_entry["doc_id"] = doc_id
            if version == 0:
                version = _ver_num
                meta_entry["version"] = version

        # 이전 버전 문서 제거
        del_expr = f"doc_id == '{doc_id}' && version < {version}"
        try:
            collection.delete(del_expr)
        except Exception:
            pass

        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            inputs = tokenizer(chunk, truncation=True, padding="longest", max_length=MAX_TOKENS, return_tensors="pt").to(device)
            with torch.no_grad():
                outs = model(**inputs)
            vec = _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
            collection.insert([[pk_counter], [vec.tolist()], [str(rel_txt)], [idx], [sec_level], [doc_id], [version]])
            pk_counter += 1

    # 컴팩트로 디스크 공간 최적화
    try:
        collection.compact()
    except Exception:
        pass

    # 메타 JSON이 수정되었을 수 있으므로 저장
    META_JSON_PATH.write_text(json.dumps(extraction_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"message": "Ingest 완료", "inserted_chunks": pk_counter}

# ----------------------------
# (NEW) 2-1) 단일 PDF 임베딩 & 인제스트
# ----------------------------
async def ingest_single_pdf(req: SinglePDFIngestRequest):
    """이미 extract_pdfs 로 추출(텍스트, 메타json) 된 상태에서 특정 PDF 한 건만 벡터 DB에 반영한다."""
    from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
    from transformers import AutoTokenizer, AutoModel
    import json, numpy as np

    MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME = "localhost", "19530", "pdf_chunks"
    MAX_TOKENS, OVERLAP = 512, 64

    if not META_JSON_PATH.exists():
        return {"error": "메타 JSON이 없습니다. 먼저 PDF 추출을 수행하세요."}

    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    # ----- 메타 키 탐색 -----
    pdf_path = Path(req.pdf_path)
    pdf_filename = pdf_path.name  # 파일명만 (PDF)
    meta_key = None
    # 1) 정확히 일치하는 키
    for k in extraction_meta:
        if k.endswith(pdf_filename):
            meta_key = k
            break
    if meta_key is None:
        # --- 메타/텍스트 미존재 → 해당 PDF 단독 추출 수행 ---
        def _extract_single(pdf_abs: Path):
            import fitz, json
            if not pdf_abs.exists():
                return None
            # security level 추정: 상위 폴더명에 securityLevelX 가 있으면 X, 없으면 1
            try:
                lvl_folder = next(p for p in pdf_abs.parents if p.name.startswith("securityLevel"))
                sec_level_val = int(lvl_folder.name.replace("securityLevel", ""))
            except StopIteration:
                sec_level_val = 1

            # 읽기
            doc = fitz.open(pdf_abs)
            pages = [p.get_text("text").strip() for p in doc]
            text_all = "\n\n".join(pages)

            # 경로 기준 상대키
            root_local = Path("local_data")
            try:
                rel_pdf = pdf_abs.relative_to(root_local)
            except ValueError:
                # pdf가 root_local 밖이면 그냥 파일명만 사용
                rel_pdf = Path(pdf_abs.name)

            txt_path_local = EXTRACTED_TEXT_DIR / rel_pdf.with_suffix(".txt")
            txt_path_local.parent.mkdir(parents=True, exist_ok=True)
            txt_path_local.write_text(text_all, encoding="utf-8")

            # doc_id/version 파싱
            stem = rel_pdf.stem
            doc_id_part, ver_num = _parse_doc_version(stem)

            info_entry = {
                "chars": len(text_all),
                "lines": len(text_all.splitlines()),
                "preview": text_all[:200].replace("\n", " ") + "…",
                "security_level": sec_level_val,
                "doc_id": doc_id_part,
                "version": ver_num,
            }

            # 메타 갱신
            if META_JSON_PATH.exists():
                meta_obj = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
            else:
                meta_obj = {}
            meta_obj[str(rel_pdf)] = info_entry
            META_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            META_JSON_PATH.write_text(json.dumps(meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            return str(rel_pdf)

        generated_key = _extract_single(pdf_path)
        if generated_key is None:
            return {"error": "PDF 경로를 찾을 수 없습니다."}
        # 다시 메타 로드
        extraction_meta = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))
        meta_key = generated_key

    txt_path = EXTRACTED_TEXT_DIR / Path(meta_key).with_suffix(".txt")
    if not txt_path.exists():
        return {"error": f"텍스트 파일이 존재하지 않습니다: {txt_path}"}

    meta_entry = extraction_meta[meta_key]
    sec_level = meta_entry["security_level"]
    doc_id = meta_entry.get("doc_id")
    version = meta_entry.get("version", 0)

    # ------------- Milvus 연결 및 스키마 확보 -------------
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection_exists = utility.has_collection(COLLECTION_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True, torch_dtype=torch.float16).to(device).eval()

    # ---------- 스키마 (필요시 생성) ----------
    def ensure_collection():
        if collection_exists:
            return Collection(COLLECTION_NAME)
        # 차원 계산
        dummy_inp = tokenizer("test", return_tensors="pt").to(device)
        with torch.no_grad():
            dummy_out = model(**dummy_inp)
        emb_dim_tmp = _mean_pooling(dummy_out, dummy_inp["attention_mask"]).shape[1]

        fields_tmp = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=emb_dim_tmp),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_idx", dtype=DataType.INT64),
            FieldSchema(name="security_level", dtype=DataType.INT64),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="version", dtype=DataType.INT64),
        ]
        schema_tmp = CollectionSchema(fields_tmp, description="PDF 청크 + 버전")
        col = Collection(name=COLLECTION_NAME, schema=schema_tmp)
        col.create_index(field_name="embedding", index_params={"metric_type": "IP", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}})
        return col

    collection = ensure_collection()
    collection.load()

    # ---------- 기존 버전 삭제 ----------
    del_expr = f"doc_id == '{doc_id}' && version < {version}"
    try:
        collection.delete(del_expr)
    except Exception:
        pass

    # ---------- 청크 & 삽입 ----------
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

    text = txt_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    pk_counter = collection.num_entities
    for idx, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, truncation=True, padding="longest", max_length=MAX_TOKENS, return_tensors="pt").to(device)
        with torch.no_grad():
            outs = model(**inputs)
        vec = _mean_pooling(outs, inputs["attention_mask"]).cpu().numpy()[0].astype("float32")
        collection.insert([[pk_counter], [vec.tolist()], [str(Path(meta_key).with_suffix('.txt'))], [idx], [sec_level], [doc_id], [version]])
        pk_counter += 1

    try:
        collection.compact()
    except Exception:
        pass

    return {"message": "단일 PDF 인제스트 완료", "doc_id": doc_id, "version": version, "chunks": len(chunks)}

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

    # 검색 (동일 doc_id 중 최신 버전만 유지)
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

# ----------------------------
# 4) DB( Milvus 컬렉션 ) 삭제
# ----------------------------
async def delete_db():
    """Milvus 인스턴스에 존재하는 모든 컬렉션을 제거"""
    from pymilvus import connections, utility

    MILVUS_HOST, MILVUS_PORT = "localhost", "19530"
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    col_names = utility.list_collections()
    for col in col_names:
        utility.drop_collection(col)
    return {"message": "삭제 완료", "dropped_collections": col_names}

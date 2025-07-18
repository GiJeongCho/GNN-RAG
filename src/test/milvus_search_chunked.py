
"""
Milvus에서 사용자 쿼리를 임베딩하여 유사도 검색 후
해당 청크의 텍스트 스니펫을 반환하는 스크립트입니다.


conda activate Qwen2.5
python /home/조기정/project/RAG_LLM/src/test/milvus_search_chunked.py  "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5
"""

import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel
# (LLM 호출용 라이브러리 예시) from transformers import AutoModelForCausalLM, pipeline

# ─────────── 설정 ───────────
MILVUS_HOST    = "localhost"
MILVUS_PORT    = "19530"
COLLECTION_NAME= "pdf_chunks"
EMBEDDING_DIR  = Path(__file__).parent / "extracted_texts"
META_JSON_PATH = EMBEDDING_DIR / "_extraction_meta.json"

# 임베딩 모델 경로 (청크 임베딩용)
EMBED_MODEL_DIR= "/home/조기정/project/RAG_LLM/src/test/embedding"
MAX_TOKENS     = 512

# LLM 모델 설정 (요약/응답 생성용)
LLM_MODEL      = "Qwen/Qwen2.5-7B-Instruct-1M"
LLM_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────

# 1) Milvus 연결 및 컬렉션 로드
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(name=COLLECTION_NAME)
collection.load()

# 2) 토크나이저·모델 준비 (임베딩)
device    = torch.device(LLM_DEVICE)
em_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_DIR, trust_remote_code=True)
em_model     = AutoModel.from_pretrained(EMBED_MODEL_DIR, trust_remote_code=True).to(device).eval()

# 3) 토크나이저·모델 준비 (LLM)
# llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
# llm_model     = AutoModelForCausalLM.from_pretrained(LLM_MODEL).to(LLM_DEVICE).eval()
# summarizer    = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, device_map="auto")

# 4) 풀링 및 청크 분할 함수

def mean_pooling(outputs, mask):
    token_emb = outputs.last_hidden_state
    mask_exp   = mask.unsqueeze(-1).expand(token_emb.size()).float()
    summed     = torch.sum(token_emb * mask_exp, dim=1)
    counts     = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
    return summed / counts


def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = 64) -> list[str]:
    ids = em_tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
    total = ids.size(0)
    chunks, start = [], 0
    while start < total:
        end = min(start + max_tokens, total)
        chunk = em_tokenizer.decode(ids[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# 5) 메타 정보 로드
with META_JSON_PATH.open("r", encoding="utf-8") as f:
    extraction_meta = json.load(f)

# 6) 검색 함수

def search(query: str, top_k: int = 5, user_level: int = 1):
    # 6.1) 쿼리 임베딩
    inputs = em_tokenizer(query, return_tensors="pt", truncation=True,
                           max_length=MAX_TOKENS, padding="longest").to(device)
    with torch.no_grad(): outputs = em_model(**inputs)
    q_emb = mean_pooling(outputs, inputs["attention_mask"])
    q_emb = F.normalize(q_emb, p=2, dim=1).cpu().numpy().astype("float32")

    # 6.2) Milvus 검색 (보안 레벨 필터링 포함)
    results = collection.search(
        data=q_emb.tolist(),
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 100}},
        limit=top_k,
        expr=f"security_level <= {user_level}",
        output_fields=["path", "chunk_idx"]
    )

    hits = []
    for hit in results[0]:
        path    = hit.entity.get("path")
        cidx    = hit.entity.get("chunk_idx")
        full_txt= (EMBEDDING_DIR / path).read_text(encoding="utf-8")
        snippet = chunk_text(full_txt)[cidx]
        hits.append({"score": hit.score, "path": path, "chunk_idx": cidx, "snippet": snippet})
    return hits

# 7) LLM 요약 함수 (예시)

def summarize_with_llm(query: str, hits: list[dict]):
    # 스니펫들을 하나의 컨텍스트로 결합
    context = "\n---\n".join([h["snippet"] for h in hits])
    prompt = f"사용자 질의: {query}\n\n관련 문서 스니펫:\n{context}\n\n위 내용을 기반으로 요약해줘."
    # 결과 생성 (예시)
    # summary = summarizer(prompt, max_length=512, do_sample=False)[0]["generated_text"]
    # return summary
    return prompt  # TODO: 실제 LLM 호출 결과 반환

# 8) CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="검색 및 요약할 질의")
    parser.add_argument("--top_k", type=int, default=5, help="검색할 상위 k개 청크")
    parser.add_argument("--level", type=int, default=1, help="사용자 보안 레벨")
    args = parser.parse_args()

    hits = search(args.query, args.top_k, args.level)
    if not hits:
        print("검색 결과가 없습니다.")
        exit(0)

    print("▶ 검색 결과 스니펫:")
    for i, h in enumerate(hits, 1):
        print(f"{i}. (score={h['score']:.4f}) {h['path']} [chunk={h['chunk_idx']}]\n{h['snippet']}\n")

    print("▶ LLM으로 요약 생성 중…")
    summary = summarize_with_llm(args.query, hits)
    print(f"\n===== 요약 결과 =====\n{summary}\n")


"""
Milvus에서 사용자 쿼리를 임베딩하여 유사도 검색 후
해당 청크의 텍스트 스니펫을 반환하는 스크립트입니다.


conda activate Qwen2.5
python /home/조기정/project/RAG_LLM/src/test/milvus_search_chunked.py  "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5
"""



import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel


# ─────────── 설정 ───────────
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COL_NAME    = "pdf_chunks"
MODEL_DIR   = "/home/조기정/project/RAG_LLM/src/test/embedding"
TEXT_DIR    = Path("extracted_texts")

MAX_TOKENS  = 512
OVERLAP     = 64    # 청크 간 중복 토큰 수
# ──────────────────────────

# Milvus 연결 & 컬렉션 로드
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
collection = Collection(COL_NAME)
collection.load()

# 토크나이저·모델 로드
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device).eval()

# 3) 청크 분할 함수 (ingest와 동일)
def chunk_text(text: str, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
    total_len = input_ids.size(0)
    chunks = []
    start = 0
    while start < total_len:
        end = min(start + max_tokens, total_len)
        chunk_ids = input_ids[start:end]
        chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        start += max_tokens - overlap
    return chunks

# mean pooling
def mean_pooling(outputs, mask):
    emb = outputs.last_hidden_state
    m   = mask.unsqueeze(-1).expand(emb.size()).float()
    sum_ = torch.sum(emb * m, dim=1)
    cnt  = torch.clamp(m.sum(dim=1), min=1e-9)
    return sum_ / cnt

# 검색 함수
def search(query:str, top_k:int=5):
    # 쿼리 임베딩
    inputs = tokenizer(query, return_tensors="pt",
                       truncation=True, max_length=MAX_TOKENS,
                       padding="longest").to(device)
    with torch.no_grad():
        out = model(**inputs)
    q_emb = mean_pooling(out, inputs["attention_mask"])
    q_emb = F.normalize(q_emb, p=2, dim=1).cpu().numpy().astype("float32")

    # Milvus 검색
    res = collection.search(
        data=q_emb.tolist(),
        anns_field="embedding",
        param={"metric_type":"IP", "params":{"ef":100}},
        limit=top_k,
        output_fields=["path","chunk_idx"]
    )

    # 결과 조립
    hits = []
    for hit in res[0]:
        path, cidx = hit.entity.get("path"), hit.entity.get("chunk_idx")
        full_text  = (TEXT_DIR / path).read_text(encoding="utf-8")
        snippet    = chunk_text(full_text)[cidx]
        hits.append({
            "id":         hit.id,
            "score":      float(hit.score),
            "path":       path,
            "chunk":      cidx,
            "text_snippet": snippet
        })
    return hits

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query",  type=str, help="검색할 문장")
    parser.add_argument("--top_k",type=int, default=5, help="상위 k개")
    args = parser.parse_args()

    results = search(args.query, args.top_k)
    for r in results:
        print(f"ID {r['id']} | score={r['score']:.4f} | path={r['path']} | chunk={r['chunk']}")
        print(f"snippet:\n{r['text_snippet']}\n{'─'*60}")

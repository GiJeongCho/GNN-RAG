
"""
백터 DB 검색 후 텍스트 스니펫 반환



사용자 쿼리를 임베딩하여 FAISS에서 유사도를 기준으로 문서 청크를 검색하고,
해당 청크의 텍스트를 함께 반환하는 스크립트입니다.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# ─────────── 설정 ───────────
MODEL_DIR   = "/home/조기정/project/RAG_LLM/src/test/embedding"
TEXT_DIR    = Path("extracted_texts")
INDEX_PATH  = Path("faiss_index.idx")
META_PATH   = Path("faiss_metadata.json")

MAX_TOKENS  = 512   # 청크당 최대 토큰 수 (ingest와 동일하게 설정)
OVERLAP     = 64    # 청크 간 중복 토큰 수 (ingest와 동일하게 설정)
# ──────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) 토크나이저·모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
model.to(device).eval()

# 2) mean pooling 정의
def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state  # (batch, seq_len, dim)
    mask_expanded    = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed           = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts           = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts  # (batch, dim)

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

# 4) FAISS 인덱스·메타데이터 로드
if not INDEX_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError("faiss_index.idx 또는 faiss_metadata.json 파일을 찾을 수 없습니다.")

index    = faiss.read_index(str(INDEX_PATH))
with META_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

# 5) 검색 함수
def search(query: str, top_k: int = 5):
    # 5.1) 쿼리 임베딩
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding="longest"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    q_emb = mean_pooling(outputs, inputs["attention_mask"])
    q_emb = F.normalize(q_emb, p=2, dim=1)
    q_vec = q_emb.cpu().numpy().astype("float32")

    # 5.2) FAISS 검색
    distances, indices = index.search(q_vec, top_k)

    # 5.3) 결과 조립
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        info      = metadata.get(str(idx), {})
        rel_path  = info.get("path")
        chunk_idx = info.get("chunk")

        # 청크 텍스트 로드
        txt_path = TEXT_DIR / rel_path
        full_text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(full_text)
        snippet = chunks[chunk_idx] if chunk_idx < len(chunks) else ""

        results.append({
            "id":         idx,
            "score":      float(score),
            "path":       rel_path,
            "chunk":      chunk_idx,
            "text_snippet": snippet
        })

    return results

# 6) CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("query", type=str, help="검색할 문장 또는 쿼리")
    parser.add_argument("--top_k", type=int, default=5, help="가져올 상위 k개 결과")
    args = parser.parse_args()

    hits = search(args.query, args.top_k)
    if not hits:
        print("▶ 검색 결과가 없습니다.")
    else:
        print(f"▶ Top {len(hits)} results for \"{args.query}\":\n")
        for r in hits:
            print(f"ID {r['id']} | score={r['score']:.4f} | path={r['path']} | chunk={r['chunk']}")
            print(f"snippet:\n{r['text_snippet']}\n{'─'*60}")

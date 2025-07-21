#!/usr/bin/env python3
# bge-m3_search_chunked.py

"""
사용자 쿼리를 BGE-M3 모델로 임베딩하여 Milvus에서 유사도가 높은 PDF 텍스트 청크를 검색하고,
각 청크의 경로, 청크 인덱스, 보안 레벨, 텍스트 스니펫을 출력하는 스크립트입니다.
--security_level 옵션으로 사용자 접근 가능 최대 보안 레벨을 지정할 수 있습니다.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pymilvus import connections, Collection
from transformers import AutoTokenizer, AutoModel

# ─────────── 설정 ───────────
MILVUS_HOST     = "localhost"
MILVUS_PORT     = "19530"
COLLECTION_NAME = "pdf_chunks"
EMBED_DIR       = Path(__file__).parent / "extracted_texts"
META_JSON_PATH  = EMBED_DIR / "_extraction_meta.json"
MODEL_PATH      = Path(__file__).parent / "embedding_bge_m3"
MAX_TOKENS      = 512
# ──────────────────────────

# mean pooling for Hugging Face outputs
def mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts

# 청크 분할 함수 (검색 결과 스니펫 재구성용)
def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = 0) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

# 검색 함수
def search(query: str, top_k: int, user_level: int):
    # 1) 메타 로드
    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding='utf-8'))

    # 2) Milvus 연결 & 컬렉션 로드
    connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 3) 토크나이저 & 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16
    ).to(device).eval()

    # 4) 쿼리 임베딩
    inputs = tokenizer(
        query, truncation=True, padding='longest', max_length=MAX_TOKENS,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    q_emb = mean_pooling(outputs, inputs['attention_mask'])
    q_vec = F.normalize(q_emb, p=2, dim=1).cpu().numpy().astype('float32')

    # 5) 필터 표현식 지정
    expr = f"security_level <= {user_level}"

    # 6) Milvus 검색
    results = collection.search(
        data=q_vec.tolist(),
        anns_field='embedding',
        param={'metric_type':'IP', 'params':{'ef':100}},
        limit=top_k,
        expr=expr,
        output_fields=['path','chunk_idx','security_level']
    )

    # 7) 결과 리스트로 반환
    hits = results[0]
    hit_list = []
    for hit in hits:
        eid = hit.id
        score = hit.score
        fields = hit.entity
        rel_txt = fields.get('path')
        cidx = fields.get('chunk_idx')
        sec_level = fields.get('security_level')

        # 원본 텍스트에서 해당 청크 재구성
        full_text = (EMBED_DIR / rel_txt).read_text(encoding='utf-8')
        chunks = chunk_text(full_text)
        snippet = chunks[cidx] if cidx < len(chunks) else ''

        hit_list.append({
            "score": score,
            "path": rel_txt,
            "chunk_idx": cidx,
            "security_level": sec_level,
            "snippet": snippet
        })
    return hit_list

# LLM 프롬프트 생성 함수

def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n---\n".join([h['snippet'] for h in hits])
    return f"사용자 질의: {query}\n\n관련 문서 스니펫:\n{context}\n\n위 내용을 참고하여 응답해 주세요."

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('query', help='검색할 문장')
    parser.add_argument('--top_k', type=int, default=5, help='상위 k개 결과')
    parser.add_argument('--user_level', type=int, required=True, help='접근 가능한 최대 보안 레벨')
    args = parser.parse_args()
    hits = search(args.query, args.top_k, args.user_level)
    if not hits:
        print("검색 결과가 없습니다.")
        exit(0)
    print("▶ 검색 스니펫:")
    for i, h in enumerate(hits, 1):
        print(f"{i}. (score={h['score']:.4f}) {h['path']} [chunk={h['chunk_idx']}] (level={h['security_level']})")
        print(h['snippet'], "\n")
    prompt = build_prompt(args.query, hits)
    print("▶ LLM 프롬프트:\n", prompt)

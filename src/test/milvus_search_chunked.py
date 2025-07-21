#!/usr/bin/env python3
# bge-m3_search_chunked.py

#!/usr/bin/env python3
# milvus_search_chunked.py

"""
Milvus에서 사용자 쿼리를 BGE-M3 임베딩으로 검색하여 상위 k개 청크를 가져오고,
LLM에 전달할 스니펫 포함 프롬프트를 생성합니다.
--level 옵션으로 접근 가능한 최대 보안 레벨을 지정하여, 그 레벨 이하의 문서만 검색됩니다.
"""

import argparse
import json
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from pymilvus import connections, Collection

# ─────────── 설정 ───────────
MILVUS_HOST    = "localhost"
MILVUS_PORT    = "19530"
COLLECTION_NAME= "pdf_chunks"
EMBED_DIR      = Path(__file__).parent / "extracted_texts"
META_JSON_PATH = EMBED_DIR / "_extraction_meta.json"
MAX_TOKENS     = 512
MODEL_DIR      = Path(__file__).parent / "embedding_Qwen4b"

OVERLAP = 64 # 텍스트 청크 분할할
# ──────────────────────────

# 텍스트 청크 분할 함수



# 검색 함수
def search(query: str, top_k: int = 5, user_level: int = 1):
    # 1) 메타 정보 로드
    meta = json.loads(META_JSON_PATH.read_text(encoding='utf-8'))

    # 2) Milvus 연결 및 컬렉션 로드
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    # 3) BGE-M3 임베딩
    # embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    # out = embed_model.encode([query], max_length=MAX_TOKENS)
    # q_emb = out['dense_vecs'][0].astype('float32')
    # 3) HuggingFace AutoModel 로 쿼리 임베딩 (Ingest 쪽과 똑같은 파이프라인)
    from transformers import AutoTokenizer, AutoModel
    import torch, torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True
    )


    def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> list[str]:
        ids       = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")[0]
        total_len = ids.size(0)
        chunks    = []
        start     = 0
        while start < total_len:
            end    = min(start + max_tokens, total_len)
            sub_ids= ids[start:end]
            chunk  = tokenizer.decode(sub_ids, skip_special_tokens=True)
            chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    model = AutoModel.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16
    ).to(device).eval()

    # mean-pooling 정의 (Ingest 쪽과 동일!)
    def mean_pooling(outputs, mask):
        token_emb = outputs.last_hidden_state
        mask_exp   = mask.unsqueeze(-1).expand(token_emb.size()).float()
        summed     = torch.sum(token_emb * mask_exp, dim=1)
        counts     = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
        return summed / counts

    # Tokenize & Encode
    inputs = tokenizer(
        query,
        truncation=True,
        padding="longest",
        max_length=MAX_TOKENS,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    q_emb = mean_pooling(outputs, inputs["attention_mask"])
    q_emb = F.normalize(q_emb, p=2, dim=1).cpu().numpy()[0].astype("float32")


    # 4) Milvus 검색: security_level 필터 사용
    expr = f"security_level <= {user_level}"
    results = collection.search(
        data=[q_emb.tolist()],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"ef": 100}},
        limit=top_k,
        expr=expr,
        output_fields=["path", "chunk_idx", "security_level"]
    )

    # 5) 검색 결과 정리
    hits = []
    for hit in results[0]:
        path = hit.entity.get("path")
        cidx = hit.entity.get("chunk_idx")
        sec_level = hit.entity.get("security_level")
        full_txt = (EMBED_DIR / path).read_text(encoding='utf-8')
        snippet = chunk_text(full_txt)[cidx]
        hits.append({
            "score": hit.score,
            "path": path,
            "chunk_idx": cidx,
            "security_level": sec_level,
            "snippet": snippet
        })
    return hits

# LLM 프롬프트 생성 함수
def build_prompt(query: str, hits: list[dict]) -> str:
    context = "\n---\n".join([h['snippet'] for h in hits])
    return f"사용자 질의: {query}\n\n관련 문서 스니펫:\n{context}\n\n위 내용을 참고하여 응답해 주세요."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="검색 및 요약할 질의")
    parser.add_argument("--top_k", type=int, default=5, help="상위 k개 청크")
    parser.add_argument("--level", type=int, default=1, dest="user_level",
                        help="접근 가능한 최대 보안 레벨")
    args = parser.parse_args()

    hits = search(args.query, top_k=args.top_k, user_level=args.user_level)
    if not hits:
        print("검색 결과가 없습니다.")
        exit(0)

    print("▶ 검색 스니펫:")
    for i, h in enumerate(hits, 1):
        print(f"{i}. (score={h['score']:.4f}) {h['path']} [chunk={h['chunk_idx']}] (level={h['security_level']})")
        print(h['snippet'], "\n")

    prompt = build_prompt(args.query, hits)
    print("▶ LLM 프롬프트:\n", prompt)

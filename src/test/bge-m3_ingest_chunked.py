#!/usr/bin/env python3
# bge-m3_ingest_chunked.py

"""
PDF 텍스트 청크(.txt)를 512토큰 이하로 분할해 BGE-M3 모델로 임베딩하고,
`_extraction_meta.json`에 기록된 보안 레벨을 함께 Milvus 컬렉션에 저장합니다.
Hugging Face Transformers를 이용해 싱글 프로세스로 임베딩합니다.
"""

import json
from pathlib import Path
import numpy as np
import torch
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from transformers import AutoTokenizer, AutoModel

# ─────────── 설정 ───────────
MILVUS_HOST     = "localhost"
MILVUS_PORT     = "19530"
COLLECTION_NAME = "pdf_chunks"
EMBED_DIR       = Path(__file__).parent / "extracted_texts"
META_JSON_PATH  = EMBED_DIR / "_extraction_meta.json"
MODEL_PATH      = Path(__file__).parent / "embedding_bge_m3"
MAX_TOKENS      = 512
OVERLAP         = 64
# ──────────────────────────

# mean pooling for Hugging Face outputs
def mean_pooling(outputs, mask):
    token_embeddings = outputs.last_hidden_state  # (batch, seq_len, dim)
    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts

# 청크 분할 함수
def chunk_text(text: str, max_tokens: int = MAX_TOKENS, overlap: int = OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

if __name__ == '__main__':
    # 1) 메타 로드
    extraction_meta = json.loads(META_JSON_PATH.read_text(encoding='utf-8'))

    # 2) Milvus 연결 & 초기화
    connections.connect(alias='default', host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # 3) Tokenizer & Model 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        str(MODEL_PATH), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.float16
    ).to(device).eval()

    # 임베딩 차원 확인
    dummy = tokenizer("test", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**dummy)
    emb_dim = mean_pooling(out, dummy['attention_mask']).shape[1]

    # 4) Milvus 스키마 생성
    fields = [
        FieldSchema(name='pk',             dtype=DataType.INT64,        is_primary=True),
        FieldSchema(name='embedding',      dtype=DataType.FLOAT_VECTOR, dim=emb_dim),
        FieldSchema(name='path',           dtype=DataType.VARCHAR,      max_length=500),
        FieldSchema(name='chunk_idx',      dtype=DataType.INT64),
        FieldSchema(name='security_level', dtype=DataType.INT64),
    ]
    schema = CollectionSchema(fields, description='PDF 청크 + 보안레벨 - BGE-M3')
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(
        field_name='embedding',
        index_params={
            'metric_type':'IP',
            'index_type':'HNSW',
            'params':{'M':16,'efConstruction':200}
        }
    )
    collection.load()

    # 5) 인제스트 루프
    pk_counter = 0
    for txt_path in EMBED_DIR.rglob('*.txt'):
        rel_txt = txt_path.relative_to(EMBED_DIR)
        rel_pdf = rel_txt.with_suffix('.pdf').as_posix()

        # 메타 키 매칭
        if rel_pdf in extraction_meta:
            meta_key = rel_pdf
        else:
            filename = rel_pdf.split('/')[-1]
            matches = [k for k in extraction_meta if k.endswith('/'+filename)]
            if matches:
                meta_key = matches[0]
            else:
                print(f"⚠️ 메타 누락: {rel_pdf}")
                continue
        sec_level = extraction_meta[meta_key]['security_level']

        text = txt_path.read_text(encoding='utf-8')
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            # 토크나이즈 & 모델
            inputs = tokenizer(
                chunk, truncation=True, padding='longest', max_length=MAX_TOKENS,
                return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs['attention_mask'])
            vec = emb.cpu().numpy()[0].astype('float32')

            # embedding must be wrapped in a list-of-lists
            collection.insert([
                [pk_counter],
                [vec.tolist()],
                [str(rel_txt)],
                [idx],
                [sec_level],
            ])
            print(f"Inserted pk={pk_counter}, path={rel_txt}, chunk={idx}, level={sec_level}")
            pk_counter += 1

    print(f"✅ ingest complete: 총 {pk_counter}개 청크 삽입되었습니다.")

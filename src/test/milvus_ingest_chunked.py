#!/usr/bin/env python3
# milvus_ingest_chunked.py

"""
PDF 텍스트 청크(.txt)를 512토큰 이하로 분할해 임베딩하고,
`_extraction_meta.json`에 기록된 보안 레벨을 함께 Milvus 컬렉션에 저장합니다.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoTokenizer, AutoModel

# ─────────── 설정 ───────────
MILVUS_HOST    = "localhost"
MILVUS_PORT    = "19530"
COLLECTION_NAME= "pdf_chunks"
EMBEDDING_DIR  = Path(__file__).parent / "extracted_texts"
META_JSON_PATH = EMBEDDING_DIR / "_extraction_meta.json"
MODEL_DIR      = "/home/조기정/project/RAG_LLM/src/test/embedding"
MAX_TOKENS     = 512  # 토큰 최대치
OVERLAP        = 64   # 청크 간 중복 토큰 수
# ──────────────────────────

# 1) 메타 정보 로드
with META_JSON_PATH.open("r", encoding="utf-8") as f:
    extraction_meta = json.load(f)

# 2) Milvus 연결 및 컬렉션 생성/로드
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

# 기존 컬렉션이 있으면 삭제
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# 토크나이저·모델 준비
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model     = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device).eval()
emb_dim   = model.config.hidden_size

# 컬렉션 스키마 정의
fields = [
    FieldSchema(name="pk",            dtype=DataType.INT64,        is_primary=True),
    FieldSchema(name="embedding",     dtype=DataType.FLOAT_VECTOR, dim=emb_dim),
    FieldSchema(name="path",          dtype=DataType.VARCHAR,      max_length=500),
    FieldSchema(name="chunk_idx",     dtype=DataType.INT64),
    FieldSchema(name="security_level",dtype=DataType.INT64),
]
schema     = CollectionSchema(fields, description="PDF 텍스트 청크와 보안레벨 저장")
collection = Collection(name=COLLECTION_NAME, schema=schema)
# HNSW 인덱스 생성
collection.create_index(
    field_name="embedding",
    index_params={"metric_type":"IP", "index_type":"HNSW", "params": {"M":16, "efConstruction":200}}
)
collection.load()

# 3) mean pooling 정의
def mean_pooling(outputs, attention_mask):
    token_emb = outputs.last_hidden_state
    mask_exp   = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    summed     = torch.sum(token_emb * mask_exp, dim=1)
    counts     = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
    return summed / counts

# 4) 텍스트 청크 분할 함수
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

# 5) 인제스트 루프
pk_counter = 0
for txt_file in EMBEDDING_DIR.rglob("*.txt"):
    # 상대경로 복원
    rel_txt = txt_file.relative_to(EMBEDDING_DIR)        # e.g. securityLevel1/foo.txt
    rel_pdf = rel_txt.with_suffix(".pdf").as_posix()    # e.g. securityLevel1/foo.pdf

    # 메타에 없으면 건너뜀
    if rel_pdf not in extraction_meta:
        print(f"⚠️ 메타 정보 누락, 건너뜁니다: {rel_pdf}")
        continue
    security_level = extraction_meta[rel_pdf]["security_level"]

    # 파일 내용 읽고 청크화
    raw_text = txt_file.read_text(encoding="utf-8")
    chunks   = chunk_text(raw_text)

    # 청크별 임베딩 및 Milvus 삽입
    for idx, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True,
                           max_length=MAX_TOKENS, padding="longest").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1).cpu().numpy().astype("float32")

        # Milvus 삽입
        collection.insert([
            [pk_counter],               # pk
            emb.tolist(),               # embedding
            [str(rel_txt)],             # path
            [idx],                      # chunk index
            [security_level],           # security level
        ])
        print(f"Inserted pk={pk_counter}, path={rel_txt}, chunk={idx}, level={security_level}")
        pk_counter += 1

print(f"✅ Milvus 컬렉션 '{COLLECTION_NAME}'에 총 {pk_counter}개 문서 삽입 완료.")

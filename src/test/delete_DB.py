#!/usr/bin/env python3
from pymilvus import connections, utility

connections.connect(alias="default", host="localhost", port="19530")

# Milvus에 있는 모든 컬렉션 이름 가져오기
all_cols = utility.list_collections()

for col in all_cols:
    utility.drop_collection(col)
    print(f"✅ 컬렉션 '{col}' 삭제 완료")

if not all_cols:
    print("⚠️ 삭제할 컬렉션이 없습니다.")



connections.connect(alias="default", host="localhost", port="19530")
if utility.has_collection("pdf_chunks_bge_m3"):
    utility.drop_collection("pdf_chunks_bge_m3")
    print("✅ 컬렉션 삭제 완료")
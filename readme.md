cd /home/조기정/project/RAG_LLM/src/test
conda activate Qwen2.5


pdf_text_extractor.py 를 실행시켜 pdf 파일의 텍스트를 긁어옵니다..
milvus_ingest_chunked.py 를 실행시켜 하위 경로의 모든 pdf 파일을 백터 DB로 저장합니다
milvus_search_chunked.py 를 실행시켜 사용자 접근 레벨 수준에 맞는 파일들을 찾습니다.

# bge-m3 임베딩 모델 이용 시

python pdf_text_extractor.py
python milvus_ingest_chunked.py
python milvus_search_chunked.py "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5 --level 3




# bge-m3 임베딩 모델 이용 시
python pdf_text_extractor.py
python bge-m3_ingest_chunked.py
python bge-m3_search_chunked.py "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5 --max_level 3
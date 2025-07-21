##### 사용법 #####

-- fast api -- (docker -compose 로 변경사항 작업하는것 대신 이렇게)
# cd /home/조기정/project/RAG_LLM/
# conda activate Qwen2.5
# uvicorn src.v1.api:app --reload | http://127.0.0.1:8000/v1/RGA_/docs 

해두고 작업 가능

-- streamlit -- (사용자 간편 테스트 )
# cd /home/조기정/project/RAG_LLM/
# uvicorn src.v1.api:app --reload
# conda activate Qwen2.5
# streamlit run app.py

-- git 작업 갱신 --
# git add .
# git commit -m "✅better than yesterday" 
# git push --force origin master
# git push --force origin qwen

####### test 경로 #######

# cd /home/조기정/project/RAG_LLM/src/test
# conda activate Qwen2.5


# pdf_text_extractor.py 를 실행시켜 pdf 파일의 텍스트를 긁어옵니다..
# milvus_ingest_chunked.py 를 실행시켜 하위 경로의 모든 pdf 파일을 백터 DB로 저장합니다
# milvus_search_chunked.py 를 실행시켜 사용자 접근 레벨 수준에 맞는 파일들을 찾습니다.

# Qwen 임베딩 모델 이용 시 https://huggingface.co/Qwen/Qwen3-Embedding-4B/tree/main

# python pdf_text_extractor.py
# python milvus_ingest_chunked.py
# python milvus_search_chunked.py "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5 --level 3

# DB 삭제 (다른 임베딩 모델 사용시 변환 필요)
# python delete_DB.py
 
# bge-m3 임베딩 모델 이용 시 https://huggingface.co/BAAI/bge-m3
# python pdf_text_extractor.py
# python bge-m3_ingest_chunked.py
# python bge-m3_search_chunked.py "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5 --user_level 3
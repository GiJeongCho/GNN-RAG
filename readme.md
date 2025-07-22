row data 사용 규칙

현재 : pdf 파일만 적용 가능
1. local_data와 같이 하위 폴더에 securityLevel"n"을 나눠서 파일로 정리한다.
2. 새 PDF를 local_data/securityLevelX/ 아래에 넣는다.
    - 업데이트 된 파일 몇개만 올릴 경우에 _ 이후에 날짜로 최신 데이터 갱신(새로운 파일 업데이트 시 해당 파일만 업데이트 하게 구성 했기 떄문)

ex )
local_data/securityLevel1
local_data/securityLevel2

local_data/securityLevel2/연구용역_관리내규_20240625.pdf
local_data/securityLevel2/청렴윤리경영_운영규정_20230731.pdf

local_data/securityLevel3
local_data/securityLevel3/81._부정청탁및금품등수수의신고사무처리에관한내규_20191128.pdf
local_data/securityLevel3/임원및직원퇴직금규정_20231204.pdf

ex )local_data/securityLevel3/임원및직원퇴직금규정_20231204.pdf


##### 사용법 #####


-- fast api -- (docker -compose 로 변경사항 작업하는것 대신 이렇게)
cd /home/조기정/project/RAG_LLM/
conda activate Qwen2.5
uvicorn src.v1.api:app --reload | http://127.0.0.1:8000/v1/RGA_/docs 

해두고 작업 가능

-- streamlit -- (사용자 간편 테스트 )
cd /home/조기정/project/RAG_LLM/
uvicorn src.v1.api:app --reload
conda activate Qwen2.5
streamlit run app.py

-- git 작업 갱신 --
git add .
git commit -m "✅better than yesterday" 
git push --force origin master
git push --force origin qwen

#### 참고 #### (start)
* 보안레벨은 1 보다 3이 높다

- 데이터 추가 시 -
local_data 폴더 → extract_pdfs 로 텍스트(.txt) & 메타(JSON) 생성 → v1/resources/extracted_texts/ 저장
Milvus 벡터 DB는 ingest_* 단계에서 .txt 를 읽어 임베딩 후 저장 (텍스트 파일은 DB에 저장되지 않음)

단일 pdf 인젝션 시, 텍스트 파일로 남기지 않고 바로 백터 DB로 저장.
**@ 나중에 텍스트 파일로 남기는 부분은 다 삭제하기.(pdf 파일 텍스트 추출 성능 검증용임)
** 단일 파일 올릴 경우에는 날짜까지 고려해서 최신만 남게 중복이 제거되는데 다중 파일 올릴때는 같은이름 날짜가 다른것은 제거가 되지 않음. [아예 같은 이름일 경우만 됨][회의록 같은 경우는 최신화가 아니라 과거 데이터도 필요할 수 있음.]

- 지원 확장자명 - 
.mbox, .pdf, .html, .txt, .md, .org, .adoc, .rst
.docx, .pptx, .odt, .odp, .xlsx, csv

#### 참고 #### (over)


### 기본 구성 ###
"""
리눅스 docker 설치 
sudo wget -qO- http://get.docker.com/
sudo apt-get update
sudo apt-get install docker.io
sudo ln -sf /usr/bin/docker.io /usr/local/bin/docker
"""

"""
docker에 milvus 설치 

Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

Start the Docker container
bash standalone_embed.sh start



 => http://127.0.0.1:9091/
"""


"""
FAISS 백터 DB 구축
!pip install faiss-cpu
! pip install sentence_transformers
! pip install --upgrade --force-reinstall sentence-transformers
! pip install pandas
! pip install pyarrow
! pip install dill
! pip install aiohttp
! pip install numpy
! pip install accelerate
"""

"""
milvus 백터 DB 구축
python3 -m pip install pymilvus==2.6.0b0
pip install "pymilvus[model]"

python /home/조기정/project/RAG_LLM/src/test/milvus_ingest_chunked.py
"""
### 기본 구성 ###


####### test 경로 #######

cd /home/조기정/project/RAG_LLM/src/test
conda activate Qwen2.5


pdf_text_extractor.py 를 실행시켜 pdf 파일의 텍스트를 긁어옵니다..
milvus_ingest_chunked.py 를 실행시켜 하위 경로의 모든 pdf 파일을 백터 DB로 저장합니다
milvus_search_chunked.py 를 실행시켜 사용자 접근 레벨 수준에 맞는 파일들을 찾습니다.

# Qwen 임베딩 모델 이용 시 https://huggingface.co/Qwen/Qwen3-Embedding-4B/tree/main

python pdf_text_extractor.py
python milvus_ingest_chunked.py
python milvus_search_chunked.py "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5 --level 3

# DB 삭제 (다른 임베딩 모델 사용시 변환 필요)
python delete_DB.py

# bge-m3 임베딩 모델 이용 시 https://huggingface.co/BAAI/bge-m3
python pdf_text_extractor.py
python bge-m3_ingest_chunked.py
python bge-m3_search_chunked.py "인사규정 제21조부터 제24조까지 요약해줘." --top_k 5 --user_level 3
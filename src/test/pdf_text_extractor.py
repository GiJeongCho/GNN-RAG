
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
#!/usr/bin/env python3
# pdf_text_extractor.py

from pathlib import Path
import fitz           # PyMuPDF
import json
from tqdm import tqdm  # 진행률 표시용

# ─────────── 설정 ───────────
# 로컬에 분류된 PDF들이 있는 디렉터리 (securityLevel1, securityLevel2, ...)
ROOT_PDF_DIR   = Path("local_data")
# 추출된 텍스트(.txt)를 저장할 디렉터리
OUTPUT_TXT_DIR = Path("extracted_texts")
# 추출 메타정보(JSON): 글자 수, 줄 수, 프리뷰, 보안레벨 등
META_JSON_PATH = OUTPUT_TXT_DIR / "_extraction_meta.json"
# ────────────────────────────

# 출력 디렉터리 생성
OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """한 PDF(멀티 페이지)의 텍스트를 전부 이어붙여 반환"""
    doc  = fitz.open(pdf_path)
    text = []
    for page in doc:
        # 순수 텍스트 추출
        page_text = page.get_text("text")
        text.append(page_text.strip())
    return "\n\n".join(text)


def main():
    # 이전에 완료한 PDF 메타 로드 (이미 추출된 항목 건너뛰기)
    done_files = {}
    if META_JSON_PATH.exists():
        done_files = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    new_meta = {}

    # 모든 PDF 탐색
    pdf_paths = list(ROOT_PDF_DIR.rglob("*.pdf"))
    if not pdf_paths:
        print("처리할 PDF가 없습니다.")
        return

    for pdf_path in tqdm(pdf_paths, desc="PDF 전처리"):
        # local_data/securityLevelX/... 에서 ROOT_PDF_DIR 기준 상대경로 계산
        pdf_rel  = pdf_path.relative_to(ROOT_PDF_DIR)
        # txt 저장 경로
        txt_path = OUTPUT_TXT_DIR / pdf_rel.with_suffix(".txt")

        # 이미 추출된 파일이면 건너뜀
        key = str(pdf_rel)
        if key in done_files and txt_path.exists():
            new_meta[key] = done_files[key]
            continue

        try:
            # PDF -> 텍스트
            pdf_text = extract_text_from_pdf(pdf_path)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(pdf_text, encoding="utf-8")

            # 상위 폴더명으로부터 보안레벨 파싱 (securityLevel2 -> 2)
            level_folder   = pdf_rel.parts[0]
            security_level = int(level_folder.replace("securityLevel", ""))

            # 검증용 메타 정보 생성
            lines = pdf_text.splitlines()
            info = {
                "chars":          len(pdf_text),
                "lines":          len(lines),
                "preview":        pdf_text[:200].replace("\n", " ") + "…",
                "security_level": security_level
            }
            new_meta[key] = info

            # 콘솔 출력
            print(f"\n✅ [{pdf_rel}] → {info['chars']} chars, {info['lines']} lines, level={security_level}")
            print(f"   preview: {info['preview']}\n")

        except Exception as e:
            print(f"❌ [{pdf_rel}] 추출 실패: {e}")

    # 메타JSON 갱신 저장
    META_JSON_PATH.write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("\n📝 전처리 완료 – 결과는 extracted_texts/ 폴더와 _extraction_meta.json을 확인하세요.")


if __name__ == "__main__":
    main()

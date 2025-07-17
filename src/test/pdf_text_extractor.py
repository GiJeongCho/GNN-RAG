from pathlib import Path
import fitz           # PyMuPDF
import json
from tqdm import tqdm  # 진행률 표시용 (선택)

ROOT_PDF_DIR   = Path("data")
OUTPUT_TXT_DIR = Path("extracted_texts")
META_JSON_PATH = Path("extracted_texts/_extraction_meta.json")  # 진행 내역‧검증용

OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """한 PDF(멀티 페이지)의 텍스트를 전부 이어붙여 반환"""
    doc  = fitz.open(pdf_path)
    text = []
    for page in doc:                        # 페이지 순회
        page_text = page.get_text("text")   # layout 無, 순수 텍스트
        text.append(page_text.strip())
    return "\n\n".join(text)

def main():
    # 이전에 완료한 PDF는 건너뛰기 위해 메타 파일 로드
    done_files = {}
    if META_JSON_PATH.exists():
        done_files = json.loads(META_JSON_PATH.read_text(encoding="utf-8"))

    new_meta = {}

    pdf_paths = list(ROOT_PDF_DIR.rglob("*.pdf"))
    if not pdf_paths:
        print("처리할 PDF가 없습니다.")
        return

    for pdf_path in tqdm(pdf_paths, desc="PDF 전처리"):
        pdf_rel  = pdf_path.relative_to(ROOT_PDF_DIR)             # data 하위 상대경로
        txt_path = OUTPUT_TXT_DIR / pdf_rel.with_suffix(".txt")   # .txt 경로 매핑

        # 이미 추출 완료된 파일이면 건너뜀
        if str(pdf_rel) in done_files and txt_path.exists():
            new_meta[str(pdf_rel)] = done_files[str(pdf_rel)]
            continue

        # 추출
        try:
            pdf_text = extract_text_from_pdf(pdf_path)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(pdf_text, encoding="utf-8")

            # 간단 검증: 글자 수/줄 수 저장
            lines = pdf_text.splitlines()
            info = {
                "chars": len(pdf_text),
                "lines": len(lines),
                "preview": pdf_text[:200].replace("\n", " ") + "…"  # 앞 200자
            }
            new_meta[str(pdf_rel)] = info

            # 콘솔에도 일부 확인
            print(f"\n✅ [{pdf_rel}] → {info['chars']} chars, {info['lines']} lines")
            print(f"   preview: {info['preview']}\n")

        except Exception as e:
            print(f"❌ [{pdf_rel}] 추출 실패: {e}")

    # 메타정보 저장(누적)
    META_JSON_PATH.write_text(json.dumps(new_meta, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print("\n📝 전처리 완료 – 결과는 extracted_texts/ 폴더와 _extraction_meta.json을 확인하세요.")

if __name__ == "__main__":
    main()
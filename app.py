import streamlit as st
import requests

# 기본 API URL (필요에 따라 수정 가능)
DEFAULT_API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="RGA_LLM RAG 대시보드", layout="centered")

st.title("📄🔎 RGA_LLM – FastAPI 연동 테스트")

# ----------------------------------
# 사이드바: API URL 설정
# ----------------------------------
with st.sidebar:
    st.header("⚙️ API 설정")
    api_base = st.text_input("FastAPI Base URL", DEFAULT_API_BASE)
    st.markdown("예: `http://127.0.0.1:8000`")
    model_choice = st.selectbox("임베딩 모델", options=["qwen", "bge_m3"], index=0)

# 헬퍼 함수
def post(endpoint: str, payload: dict | None = None):
    url = f"{api_base}{endpoint}"
    try:
        r = requests.post(url, json=payload, timeout=60)
        return r
    except requests.exceptions.RequestException as e:
        st.error(f"요청 실패: {e}")
        return None

st.write("""이 대시보드는 FastAPI 서버의 RAG 엔드포인트를 순차적으로 호출하여
1) PDF 추출 → 2) 임베딩·Ingest → 3) 검색 과정을 손쉽게 테스트할 수 있도록 합니다.""")

# ==================================
# 1) PDF → 텍스트 추출
# ==================================
st.subheader("① PDF 추출")
with st.expander("PDF 추출 설정", expanded=True):
    dir_path = st.text_input("PDF 루트 디렉터리 경로", placeholder="/absolute/path/to/pdf_root")
    if st.button("🚀 추출 실행", key="extract"):
        if not dir_path:
            st.warning("경로를 입력하세요.")
        else:
            with st.spinner("PDF 텍스트 추출 중..."):
                resp = post("/v1/rag/extract", {"dir_path": dir_path})
            if resp is not None:
                st.json(resp.json() if resp.ok else resp.text)

st.divider()

# ==================================
# 2) Milvus 임베딩 Ingest
# ==================================
st.subheader("② 임베딩 & Milvus Ingest")
col_ing1, col_ing2 = st.columns(2)
with col_ing1:
    if st.button("🔄 전체 Ingest", key="ingest_all"):
        with st.spinner("임베딩·Ingest 진행 중..."):
            resp = post(f"/v1/rag/ingest?model={model_choice}")
        if resp is not None:
            st.json(resp.json() if resp.ok else resp.text)
with col_ing2:
    single_path = st.text_input("단일 PDF 절대경로", key="single_path")
    if st.button("➕ 단일 파일 추가(파일 업데이트 시 이전 파일 이름과 _ 이후에 날짜로 최신 데이터 갱신)", key="ingest_single"):
        if not single_path:
            st.warning("PDF 경로를 입력하세요.")
        else:
            with st.spinner("단일 PDF 인제스트 중..."):
                resp = post(f"/v1/rag/ingest-file?model={model_choice}", {"pdf_path": single_path})
            if resp is not None:
                st.json(resp.json() if resp.ok else resp.text)

st.divider()

# ==================================
# 3) 벡터 검색
# ==================================
st.subheader("③ 검색")
query = st.text_input("검색 질문", placeholder="예) 인사규정 제21조부터 제24조까지 요약해줘.")
col1, col2 = st.columns(2)
with col1:
    top_k = st.number_input("top_k", min_value=1, max_value=50, value=5, step=1)
with col2:
    user_level = st.number_input("사용자 보안 레벨", min_value=1, max_value=10, value=1, step=1)

# 특정 문서 필터
with st.expander("🔍 특정 문서만 검색", expanded=False):
    doc_names_raw = st.text_area("문서 이름 목록 (콤마/줄바꿈 구분)", placeholder="example.pdf, sample_20240625.pdf")
    doc_names = [d.strip() for d in doc_names_raw.replace("\n", ",").split(",") if d.strip()]

if st.button("🔍 검색 실행", key="search"):
    if not query.strip():
        st.warning("질문을 입력하세요.")
    else:
        base_payload = {"query": query, "top_k": top_k, "user_level": user_level, "model_name": model_choice}
        if doc_names:
            base_payload["doc_names"] = doc_names
            endpoint = "/v1/rag/search-docs"
        else:
            endpoint = "/v1/rag/search"
        with st.spinner("검색 중..."):
            resp = post(endpoint, base_payload)
        if resp is not None and resp.ok:
            data = resp.json()
            st.success(f"검색 완료 – 소요 시간: {data.get('elapsed_sec', '?')}s")

            for i, h in enumerate(data.get("hits", []), 1):
                with st.expander(f"{i}. {h['path']} | score={h['score']:.4f} | level={h['security_level']}"):
                    st.write(h["snippet"])

            st.subheader("LLM 프롬프트")
            st.code(data.get("prompt", ""), language="markdown")
        elif resp is not None:
            st.error(resp.text)

st.divider()

# ==================================
# 4) 선택 문서 삭제
# ==================================
st.subheader("④ 선택 문서 삭제")
with st.expander("문서 삭제", expanded=False):
    del_doc_raw = st.text_area("삭제할 문서 이름 목록 (콤마/줄바꿈)", key="del_docs")
    del_btn = st.button("🗑️ 문서 삭제", key="delete_selected")

    if del_btn:
        del_docs = [d.strip() for d in del_doc_raw.replace("\n", ",").split(",") if d.strip()]
        if not del_docs:
            st.warning("삭제할 문서명을 입력하세요.")
        else:
            payload = {"doc_names": del_docs}
            with st.spinner("삭제 중..."):
                resp = post("/v1/rag/delete-docs", payload)
            if resp is not None:
                st.json(resp.json() if resp.ok else resp.text)

st.divider()

# ==================================
# 5) DB 삭제
# ==================================
st.subheader("🗑️ Milvus DB 삭제")
# 두 단계 확인 로직
if st.button("⚠️ 전체 컬렉션 삭제", key="delete_db"):
    st.session_state["want_delete"] = True

if st.session_state.get("want_delete"):
    st.warning("정말 모든 컬렉션을 삭제하시겠습니까?")
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("✅ 네, 삭제", key="confirm_delete_db"):
            with st.spinner("컬렉션 삭제 중..."):
                resp = post("/v1/rag/delete-db")
            if resp is not None:
                st.json(resp.json() if resp.ok else resp.text)
            st.session_state.pop("want_delete", None)
    with col_del2:
        if st.button("❌ 취소", key="cancel_delete_db"):
            st.session_state.pop("want_delete", None)





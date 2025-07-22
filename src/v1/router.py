from fastapi import APIRouter, status, Request
from .main import (
    PDFExtractRequest, RAGSearchRequest,
    extract_pdfs, ingest_embeddings, search_documents,
    ingest_single_pdf, delete_db, SinglePDFIngestRequest,
    DeleteDocsRequest, delete_selected_docs
)

router_v1 = APIRouter(
    prefix="/v1",
    tags=["rag"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"}
    },
)


# -----------------------------
# RAG: PDF 추출
# -----------------------------
@router_v1.post("/rag/extract", summary="PDF 경로를 받아 텍스트와 메타를 추출")
async def rag_extract_endpoint(req: PDFExtractRequest, request: Request):
    print(f"Extract Request from {request.client.host} -> {req.dir_path}")
    return await extract_pdfs(req)

# -----------------------------
# RAG: 임베딩 인제스트
# -----------------------------
@router_v1.post("/rag/ingest", summary="추출된 텍스트를 임베딩하여 Milvus에 저장")
async def rag_ingest_endpoint(request: Request, model: str | None = None):
    print(f"Ingest Request from {request.client.host} (model={model})")
    return await ingest_embeddings(model_name=model)

# -----------------------------
# RAG: 단일 PDF 인제스트
# -----------------------------
@router_v1.post("/rag/ingest-file", summary="단일 PDF만 벡터 DB에 반영")
async def rag_ingest_file_endpoint(req: SinglePDFIngestRequest, request: Request, model: str | None = None):
    print(f"Single Ingest from {request.client.host}: {req.pdf_path} (model={model})")
    return await ingest_single_pdf(req, model_name=model)

# -----------------------------
# RAG: 검색
# -----------------------------
@router_v1.post("/rag/search", summary="사용자 질의를 받아 벡터 검색 및 스니펫 반환")
async def rag_search_endpoint(req: RAGSearchRequest, request: Request):
    print(f"Search Request from {request.client.host}: '{req.query}' (level={req.user_level})")
    return await search_documents(req)

# -----------------------------
# RAG: DB 전체 삭제
# -----------------------------
@router_v1.post("/rag/delete-db", summary="Milvus의 모든 컬렉션 삭제")
async def rag_delete_db_endpoint(request: Request):
    print(f"Delete DB Request from {request.client.host}")
    return await delete_db()

# -----------------------------
# RAG: 선택 문서 검색
# -----------------------------
@router_v1.post("/rag/search-docs", summary="특정 문서 이름 리스트로 제한하여 검색")
async def rag_search_docs_endpoint(req: RAGSearchRequest, request: Request):
    print(f"Selective Search from {request.client.host}: docs={req.doc_names}")
    return await search_documents(req)

# -----------------------------
# RAG: 특정 문서 삭제
# -----------------------------
@router_v1.post("/rag/delete-docs", summary="문서 이름 리스트에 해당하는 벡터만 삭제")
async def rag_delete_docs_endpoint(req: DeleteDocsRequest, request: Request):
    print(f"Delete Docs from {request.client.host}: {req.doc_names} (only_single={req.only_single})")
    return await delete_selected_docs(req) 
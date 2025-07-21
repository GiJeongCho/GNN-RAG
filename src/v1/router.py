from fastapi import APIRouter, status, Request
from .main import (
    PDFExtractRequest, RAGSearchRequest,
    extract_pdfs, ingest_embeddings, search_documents,
    delete_db
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
async def rag_ingest_endpoint(request: Request):
    print(f"Ingest Request from {request.client.host}")
    return await ingest_embeddings()

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
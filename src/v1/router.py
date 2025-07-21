from fastapi import APIRouter, status, Request
from v1.main import (
    PosTypesRequest, get_pos_types,
    PDFExtractRequest, RAGSearchRequest,
    extract_pdfs, ingest_embeddings, search_documents
)

router_v1 = APIRouter(
    prefix="/v1",
    tags=["score", "rag"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"}
    },
)

# -----------------------------
# 기존 POS 타입 엔드포인트
# -----------------------------
@router_v1.post("/pos-types", summary="POS 유형 빈도 판단 [입력된 문장에서 사용된 품사의 종류와 수를 반환]")
async def pos_types_endpoint(req: PosTypesRequest, request: Request):
    print(f"POS Request received from {request.client.host}")
    return await get_pos_types(req)

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
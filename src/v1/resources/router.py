from fastapi import APIRouter, status, Request
from v1.main import (
    PosTypesRequest, get_pos_types,
)

router_v1 = APIRouter(
    prefix="/v1",
    tags=["score"],
    responses={
        status.HTTP_200_OK: {"description": "Successful Response"},
        status.HTTP_401_UNAUTHORIZED: {"description": "Unauthorized"},
        status.HTTP_403_FORBIDDEN: {"description": "Forbidden"},
        status.HTTP_404_NOT_FOUND: {"description": "Not found"}
    },
)

@router_v1.post("/pos-types", summary="POS 유형 빈도 판단 [입력된 문장에서 사용된 품사의 종류와 수를 반환]")
async def pos_types_endpoint(req: PosTypesRequest, request: Request):
    print(f"Request received from {request.client.host}")
    return await get_pos_types(req)

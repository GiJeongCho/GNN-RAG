from fastapi import FastAPI
from .router import router_v1

app = FastAPI(
    docs_url="/v1/RGA_/docs",
    redoc_url="/v1/RGA_/redoc",
    openapi_url="/v1/RGA_/openapi.json"
)

# 라우터를 등록합니다.
app.include_router(router_v1)
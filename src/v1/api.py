from fastapi import FastAPI
from v1.router import router_v1

app = FastAPI(
    docs_url="/v1/RGA_LLM/docs",
    redoc_url="/v1/RGA_LLM/redoc",
    openapi_url="/v1/RGA_LLM/openapi.json"
)

# 라우터를 등록합니다.
app.include_router(router_v1)
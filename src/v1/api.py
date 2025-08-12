from fastapi import FastAPI
from .router import router_v1
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    docs_url="/v1/RGA_/docs",
    redoc_url="/v1/RGA_/redoc",
    openapi_url="/v1/RGA_/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 라우터를 등록합니다.
app.include_router(router_v1)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
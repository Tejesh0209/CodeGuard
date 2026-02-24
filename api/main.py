from fastapi import FastAPI
from api.webhook_handler import router as webhook_router

app = FastAPI(title="Codeguard", version="0.1.0")
app.include_router(webhook_router)

@app.get("/health")
async def health():
    return {"Status":"ok","service":"codeguard"}

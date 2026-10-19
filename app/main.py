from fastapi import FastAPI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

from app.config import settings

app = FastAPI(title="Governed RAG Platform")
engine = create_engine(settings.database_url, pool_pre_ping=True)
qdrant = QdrantClient(url=settings.qdrant_url)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/deps")
def health_deps():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    qdrant.get_collections()
    return {"mysql": "ok", "qdrant": "ok"}
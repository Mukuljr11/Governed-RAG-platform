import httpx

from app.config import settings
from app.observability import tracer


def embed_texts(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    vectors: list[list[float]] = []
    with tracer.start_as_current_span("ollama.embed") as span:
        span.set_attribute("embed.model", settings.embedding_model)
        span.set_attribute("embed.texts", len(texts))
        with httpx.Client(base_url=settings.ollama_url, timeout=300) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                resp = client.post(
                    "/api/embed",
                    json={"model": settings.embedding_model, "input": batch},
                )
                resp.raise_for_status()
                vectors.extend(resp.json()["embeddings"])
    return vectors
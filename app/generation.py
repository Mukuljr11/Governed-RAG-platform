import re

import httpx

from app.config import settings

NOT_FOUND = "I couldn't find this in the provided documents."

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions using ONLY the numbered context passages provided.

Rules:
- Use only facts stated in the context. Do not use outside knowledge.
- After every sentence that uses a passage, add its number in square brackets, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: {NOT_FOUND}
- Keep the answer short and direct."""


def build_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] (source: {c['filename']}, page {c['page_number']})\n{c['content']}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> str:
    user_prompt = (
        f"Context:\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
    )
    with httpx.Client(base_url=settings.ollama_url, timeout=600) as client:
        resp = client.post(
            "/api/chat",
            json={
                "model": settings.llm_model,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    numbers: set[int] = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        numbers.update(int(x) for x in re.findall(r"\d+", group))

    sources = []
    for n in sorted(numbers):
        if 1 <= n <= len(chunks):
            c = chunks[n - 1]
            sources.append(
                {
                    "ref": n,
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "filename": c["filename"],
                    "page_number": c["page_number"],
                    "snippet": c["content"],
                }
            )
    return sources
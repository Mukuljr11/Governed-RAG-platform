from typing import Any

from mcp.server.fastmcp import FastMCP

from app.config import settings
from app.db import SessionLocal
from app.models import ApprovalRequest
from app.retrieval import hybrid_search

mcp = FastMCP("governed-rag-tools", host="0.0.0.0", port=8001)


@mcp.tool()
def search_documents(query: str, user_id: int, top_k: int = 4) -> dict[str, Any]:
    """Search the documents the given user is allowed to read. Returns the best matching passages with filename and page number."""
    db = SessionLocal()
    try:
        rows = hybrid_search(
            db, query, user_id=user_id, top_k=top_k, min_similarity=settings.min_similarity
        )
    finally:
        db.close()
    return {
        "results": [
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "filename": r["filename"],
                "page_number": r["page_number"],
                "content": r["content"],
                "similarity": round(r["similarity"], 3),
            }
            for r in rows
        ]
    }


@mcp.tool()
def create_approval_request(
    requester_id: int,
    summary: str,
    policy_reference: str,
    reviewer_id: int | None = None,
    reviewer_note: str = "",
) -> dict[str, Any]:
    """Record a request in the approvals table. Only call this after a human has approved it."""
    db = SessionLocal()
    try:
        req = ApprovalRequest(
            requester_id=requester_id,
            reviewer_id=reviewer_id,
            summary=summary,
            policy_reference=policy_reference,
            reviewer_note=reviewer_note or None,
        )
        db.add(req)
        db.commit()
        return {"request_id": req.id, "status": req.status}
    finally:
        db.close()
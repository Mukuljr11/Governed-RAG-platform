from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embeddings import embed_texts

HYBRID_SQL = text("""
WITH vec AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rnk
    FROM (
        SELECT id, embedding <=> CAST(:qvec AS vector) AS dist
        FROM chunks
        ORDER BY embedding <=> CAST(:qvec AS vector)
        LIMIT :k
    ) v
),
kw AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rnk
    FROM (
        SELECT c.id, ts_rank_cd(c.content_tsv, qq.q) AS score
        FROM chunks c,
             (SELECT CAST(
                  replace(CAST(websearch_to_tsquery('english', :qtext) AS text), '&', '|')
                  AS tsquery) AS q) qq
        WHERE c.content_tsv @@ qq.q
        ORDER BY score DESC
        LIMIT :k
    ) t
)
SELECT
    c.id AS chunk_id,
    c.document_id,
    d.filename,
    c.page_number,
    c.content,
    CAST(COALESCE(1.0 / (60 + vec.rnk), 0) + COALESCE(1.0 / (60 + kw.rnk), 0) AS float) AS score,
    CAST(1 - (c.embedding <=> CAST(:qvec AS vector)) AS float) AS similarity,
    vec.rnk AS vector_rank,
    kw.rnk AS keyword_rank
FROM vec
FULL OUTER JOIN kw ON vec.id = kw.id
JOIN chunks c ON c.id = COALESCE(vec.id, kw.id)
JOIN documents d ON d.id = c.document_id
ORDER BY score DESC
LIMIT :top_k
""")


def hybrid_search(
    db: Session,
    query: str,
    top_k: int = 5,
    candidates: int = 20,
    min_similarity: float = 0.0,
) -> list[dict]:
    qvec = embed_texts([query])[0]
    vec_literal = "[" + ",".join(str(x) for x in qvec) + "]"
    rows = db.execute(
        HYBRID_SQL,
        {"qvec": vec_literal, "qtext": query, "k": candidates, "top_k": top_k},
    ).mappings().all()
    results = [dict(r) for r in rows]
    return [
        r
        for r in results
        if r["similarity"] >= min_similarity or r["keyword_rank"] is not None
    ]
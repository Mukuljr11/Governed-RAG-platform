from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import delete

from app.db import SessionLocal
from app.embeddings import embed_texts
from app.models import Chunk, Document, IngestionJob

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)


def extract_pages(path: str, content_type: str) -> list[tuple[int, str]]:
    if content_type == "application/pdf":
        reader = PdfReader(path)
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [(1, text)]


def run_ingestion(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(IngestionJob, job_id)
        doc = db.get(Document, job.document_id)
        job.status = "processing"
        db.commit()

        pieces: list[tuple[int, str]] = []
        for page_no, text in extract_pages(doc.storage_path, doc.content_type):
            for part in splitter.split_text(text):
                if part.strip():
                    pieces.append((page_no, part))
        if not pieces:
            raise ValueError("No extractable text found (scanned PDF?)")

        vectors = embed_texts([text for _, text in pieces])

        db.execute(delete(Chunk).where(Chunk.document_id == doc.id))
        db.add_all(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                page_number=page_no,
                content=text,
                embedding=vec,
            )
            for i, ((page_no, text), vec) in enumerate(zip(pieces, vectors))
        )
        job.status = "done"
        db.commit()
    except Exception as exc:
        db.rollback()
        job = db.get(IngestionJob, job_id)
        job.status = "failed"
        job.error = str(exc)[:1000]
        db.commit()
    finally:
        db.close()
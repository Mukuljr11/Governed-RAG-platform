from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import engine, get_db
from app.ingestion import run_ingestion
from app.models import Chunk, Document, IngestionJob, User
from app.retrieval import hybrid_search
from app.generation import NOT_FOUND, extract_citations, generate_answer

app = FastAPI(title="Governed RAG Platform")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/deps")
def health_deps():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"postgres": "ok"}

class UserIn(BaseModel):
    email: str


class DocumentIn(BaseModel):
    owner_id: int
    filename: str
    content_type: str = "application/pdf"


@app.post("/users")
def create_user(body: UserIn, db: Session = Depends(get_db)):
    user = User(email=body.email)
    db.add(user)
    db.commit()
    return {"id": user.id, "email": user.email}


@app.post("/documents")
def create_document(body: DocumentIn, db: Session = Depends(get_db)):
    doc = Document(
        owner_id=body.owner_id,
        filename=body.filename,
        content_type=body.content_type,
    )
    db.add(doc)
    db.flush()
    db.add(IngestionJob(document_id=doc.id))
    db.commit()
    return {"id": doc.id, "filename": doc.filename}


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    docs = db.scalars(select(Document)).all()
    return [{"id": d.id, "owner_id": d.owner_id, "filename": d.filename} for d in docs]    

ALLOWED_TYPES = {"application/pdf", "text/plain"}


@app.post("/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    owner_id: int = Form(...),
    db: Session = Depends(get_db),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only PDF and TXT files are supported")
    if db.get(User, owner_id) is None:
        raise HTTPException(404, "Owner not found")

    doc = Document(
        owner_id=owner_id,
        filename=file.filename or "upload",
        content_type=file.content_type,
    )
    db.add(doc)
    db.flush()

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{doc.id}{Path(doc.filename).suffix.lower()}"
    path.write_bytes(file.file.read())
    doc.storage_path = str(path)

    job = IngestionJob(document_id=doc.id)
    db.add(job)
    db.commit()

    background_tasks.add_task(run_ingestion, job.id)
    return {"document_id": doc.id, "job_id": job.id, "status": "pending"}


@app.get("/documents/{doc_id}/status")
def document_status(doc_id: int, db: Session = Depends(get_db)):
    job = db.scalars(
        select(IngestionJob)
        .where(IngestionJob.document_id == doc_id)
        .order_by(IngestionJob.id.desc())
    ).first()
    if job is None:
        raise HTTPException(404, "Document not found")
    chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == doc_id))
    return {"status": job.status, "error": job.error, "chunks": chunks}    

class SearchIn(BaseModel):
    query: str
    top_k: int = 5
    min_similarity: float = 0.6


@app.post("/search")
def search(body: SearchIn, db: Session = Depends(get_db)):
    results = hybrid_search(db, body.query, body.top_k, min_similarity=body.min_similarity)
    return {"query": body.query, "results": results}

class AskIn(BaseModel):
    question: str
    top_k: int = 5


@app.post("/ask")
def ask(body: AskIn, db: Session = Depends(get_db)):
    chunks = hybrid_search(
        db, body.question, body.top_k, min_similarity=settings.min_similarity
    )
    if not chunks:
        return {
            "question": body.question,
            "answer": NOT_FOUND,
            "sources": [],
            "grounded": True,
        }

    answer = generate_answer(body.question, chunks)
    sources = extract_citations(answer, chunks)
    return {
        "question": body.question,
        "answer": answer,
        "sources": sources,
        "grounded": bool(sources) or answer.strip() == NOT_FOUND,
    }    
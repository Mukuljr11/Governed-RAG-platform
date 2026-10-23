from fastapi import Depends, FastAPI
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import engine, get_db
from app.models import Document, IngestionJob, User

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
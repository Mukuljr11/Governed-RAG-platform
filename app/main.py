import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from app.agent import run_agent
from app.config import settings
from app.db import engine, get_db
from app.generation import NOT_FOUND, extract_citations, generate_answer, has_content
from app.guardrails import GuardrailViolation, get_analyzer, redact, sanitize_input, strip_placeholders
from app.ingestion import run_ingestion
from app.models import Chunk, Document, DocumentShare, IngestionJob, User
from app.request_agent import decide, get_graph, get_request, start_request
from app.retrieval import hybrid_search
from app.security import (
    accessible_document,
    create_access_token,
    get_current_user,
    hash_password,
    require_manager,
    verify_password,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(settings.jwt_secret) < 32:
        raise RuntimeError("JWT_SECRET must be set to at least 32 characters")
    get_graph()  # create checkpoint tables before any request holds a transaction
    get_analyzer()  # load the PII model once at startup
    yield


app = FastAPI(title="Governed RAG Platform", lifespan=lifespan)


@app.exception_handler(GuardrailViolation)
async def guardrail_handler(request: Request, exc: GuardrailViolation):
    return JSONResponse(
        status_code=400, content={"detail": exc.message, "guardrail": exc.code}
    )


# ---------- health ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/deps")
def health_deps():
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return {"postgres": "ok"}


# ---------- auth ----------
class RegisterIn(BaseModel):
    email: str
    password: str


@app.post("/auth/register", status_code=201)
def register(body: RegisterIn, db: Session = Depends(get_db)):
    email = body.email.strip().lower()
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    user = User(email=email, role="employee", password_hash=hash_password(body.password))
    db.add(user)
    db.commit()
    return {"id": user.id, "email": user.email, "role": user.role}


@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    if (
        user is None
        or not user.password_hash
        or not verify_password(form.password, user.password_hash)
    ):
        raise HTTPException(401, "Incorrect email or password")
    return {"access_token": create_access_token(user.id), "token_type": "bearer"}


@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "role": user.role}


# ---------- documents ----------
ALLOWED_TYPES = {"application/pdf", "text/plain"}


class ShareIn(BaseModel):
    email: str


@app.post("/documents/upload")
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    visibility: str = Form("private"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if visibility not in {"private", "company"}:
        raise HTTPException(422, "visibility must be 'private' or 'company'")
    if visibility == "company" and user.role != "manager":
        raise HTTPException(403, "Only managers can publish company-wide documents")
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Only PDF and TXT files are supported")

    doc = Document(
        owner_id=user.id,
        filename=file.filename or "upload",
        content_type=file.content_type,
        visibility=visibility,
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
    return {
        "document_id": doc.id,
        "job_id": job.id,
        "visibility": visibility,
        "status": "pending",
    }


@app.get("/documents")
def list_documents(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    shared_ids = select(DocumentShare.document_id).where(DocumentShare.user_id == user.id)
    docs = db.scalars(
        select(Document)
        .where(
            or_(
                Document.owner_id == user.id,
                Document.visibility == "company",
                Document.id.in_(shared_ids),
            )
        )
        .order_by(Document.id)
    ).all()
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "owner_id": d.owner_id,
            "visibility": d.visibility,
        }
        for d in docs
    ]


@app.get("/documents/{doc_id}/status")
def document_status(
    doc_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    if accessible_document(db, doc_id, user.id) is None:
        raise HTTPException(404, "Document not found")
    job = db.scalars(
        select(IngestionJob)
        .where(IngestionJob.document_id == doc_id)
        .order_by(IngestionJob.id.desc())
    ).first()
    if job is None:
        raise HTTPException(404, "Document not found")
    chunks = db.scalar(select(func.count(Chunk.id)).where(Chunk.document_id == doc_id))
    return {"status": job.status, "error": job.error, "chunks": chunks}


@app.post("/documents/{doc_id}/share")
def share_document(
    doc_id: int,
    body: ShareIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = db.get(Document, doc_id)
    if doc is None or doc.owner_id != user.id:
        raise HTTPException(404, "Document not found")
    target = db.scalar(select(User).where(User.email == body.email.strip().lower()))
    if target is None:
        raise HTTPException(404, "User not found")
    existing = db.scalar(
        select(DocumentShare.id).where(
            DocumentShare.document_id == doc_id, DocumentShare.user_id == target.id
        )
    )
    if existing is None:
        db.add(DocumentShare(document_id=doc_id, user_id=target.id))
        db.commit()
    return {"document_id": doc_id, "shared_with": target.email}


# ---------- search and answers ----------
class SearchIn(BaseModel):
    query: str
    top_k: int = 5
    min_similarity: float = 0.0


class AskIn(BaseModel):
    question: str
    top_k: int = 5


@app.post("/search")
def search(
    body: SearchIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    query = sanitize_input(body.query, user.id)
    results = hybrid_search(
        db, strip_placeholders(question), query, user_id=user.id, top_k=body.top_k, min_similarity=body.min_similarity
    )
    for r in results:
        r["content"] = redact(r["content"])
    return {"query": query, "results": results}


@app.post("/ask")
def ask(body: AskIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    question = sanitize_input(body.question, user.id)
    chunks = hybrid_search(
        db,
        strip_placeholders(question),
        user_id=user.id,
        top_k=body.top_k,
        min_similarity=settings.min_similarity,
    )
    if not chunks:
        return {"question": question, "answer": NOT_FOUND, "sources": [], "grounded": True}

    answer = generate_answer(question, chunks)
    sources = extract_citations(answer, chunks)
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "grounded": (bool(sources) and has_content(answer)) or answer.strip() == NOT_FOUND,
    }


@app.post("/ask/agent")
def ask_agent(
    body: AskIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    question = sanitize_input(body.question, user.id)
    return run_agent(db, question, user.id)


# ---------- approval requests ----------
class RequestIn(BaseModel):
    request_text: str


class DecisionIn(BaseModel):
    approved: bool
    note: str = ""


@app.post("/requests")
def create_request(body: RequestIn, user: User = Depends(get_current_user)):
    request_text = sanitize_input(body.request_text, user.id)
    return start_request(user.id, request_text)


@app.get("/requests/{thread_id}")
def read_request(thread_id: str, user: User = Depends(get_current_user)):
    try:
        data = get_request(thread_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    if data["requester_id"] != user.id and user.role != "manager":
        raise HTTPException(404, "Unknown request")
    return data


@app.post("/requests/{thread_id}/decision")
def decide_request(
    thread_id: str, body: DecisionIn, manager: User = Depends(require_manager)
):
    try:
        data = get_request(thread_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    if data["requester_id"] == manager.id:
        raise HTTPException(403, "You cannot approve or reject your own request")
    try:
        return decide(thread_id, body.approved, redact(body.note), manager.id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
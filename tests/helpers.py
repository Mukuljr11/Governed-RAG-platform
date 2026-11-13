from app.config import settings
from app.models import Chunk, Document, User
from app.security import create_access_token, hash_password

_PASSWORD_HASH = hash_password("password-123")  # hash once, reuse in every test


def make_user(db, email: str, role: str = "employee") -> User:
    user = User(email=email, role=role, password_hash=_PASSWORD_HASH)
    db.add(user)
    db.commit()
    return user


def vec(*lead: float) -> list[float]:
    """A 1024-number vector whose first numbers are given and the rest are zero."""
    v = [0.0] * settings.embedding_dim
    for i, x in enumerate(lead):
        v[i] = x
    return v


def add_document(db, owner, filename, visibility="private", chunks=()):
    doc = Document(
        owner_id=owner.id,
        filename=filename,
        content_type="text/plain",
        visibility=visibility,
    )
    db.add(doc)
    db.flush()
    for i, (content, embedding) in enumerate(chunks):
        db.add(
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                page_number=1,
                content=content,
                embedding=embedding,
            )
        )
    db.commit()
    return doc


def auth(user) -> dict:
    return {"Authorization": f"Bearer {create_access_token(user.id)}"}
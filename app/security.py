from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from pwdlib import PasswordHash
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal
from app.models import Document, DocumentShare, User

_hasher = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _hasher.verify(password, hashed)


def create_access_token(user_id: int) -> str:
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    return jwt.encode(
        {"sub": str(user_id), "exp": expires}, settings.jwt_secret, algorithm="HS256"
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    unauthorized = HTTPException(
        401, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = int(payload["sub"])
    except (InvalidTokenError, KeyError, ValueError):
        raise unauthorized
    with SessionLocal() as db:  # short-lived: do not hold a connection during slow LLM calls
        user = db.get(User, user_id)
    if user is None:
        raise unauthorized
    return user


def require_manager(user: User = Depends(get_current_user)) -> User:
    if user.role != "manager":
        raise HTTPException(403, "Manager role required")
    return user


def accessible_document(db: Session, doc_id: int, user_id: int) -> Document | None:
    doc = db.get(Document, doc_id)
    if doc is None:
        return None
    if doc.owner_id == user_id or doc.visibility == "company":
        return doc
    shared = db.scalar(
        select(DocumentShare.id).where(
            DocumentShare.document_id == doc_id, DocumentShare.user_id == user_id
        )
    )
    return doc if shared else None
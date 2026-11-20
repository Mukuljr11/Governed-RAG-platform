import os

# Must be set before any app module is imported.
os.environ["POSTGRES_DB"] = "ragdb_test"
os.environ["JWT_SECRET"] = "test-secret-0123456789abcdef0123456789abcdef"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import settings
from app.db import SessionLocal, engine
from app.main import app
from tests.helpers import vec

assert settings.postgres_db.endswith("_test"), "tests must never use the dev database"


@pytest.fixture(autouse=True)
def clean_db():
    assert settings.postgres_db.endswith("_test")
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE chunks, document_shares, ingestion_jobs, approval_requests, "
                "documents, users RESTART IDENTITY CASCADE"
            )
        )
    yield


@pytest.fixture(autouse=True)
def fake_embedding(monkeypatch):
    # Every query embeds to the same vector, so tests control rankings with chunk vectors.
    monkeypatch.setattr(
        "app.retrieval.embed_texts", lambda texts, **kwargs: [vec(1.0) for _ in texts]
    )


@pytest.fixture
def db():
    with SessionLocal() as session:
        yield session


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
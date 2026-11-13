from app.models import DocumentShare
from app.retrieval import hybrid_search
from tests.helpers import add_document, make_user, vec


def test_only_the_owner_finds_a_private_chunk(db):
    alice = make_user(db, "alice@example.com")
    bob = make_user(db, "bob@example.com")
    mia = make_user(db, "mia@example.com", "manager")
    add_document(db, alice, "falcon.txt", "private", [("Project Falcon budget is 2 million dollars", vec(1.0))])

    assert hybrid_search(db, "falcon budget", user_id=alice.id)
    assert hybrid_search(db, "falcon budget", user_id=bob.id) == []
    assert hybrid_search(db, "falcon budget", user_id=mia.id) == []  # managers do not see private files


def test_sharing_grants_access_to_that_user_only(db):
    alice = make_user(db, "alice@example.com")
    bob = make_user(db, "bob@example.com")
    carol = make_user(db, "carol@example.com")
    doc = add_document(db, alice, "falcon.txt", "private", [("Project Falcon budget is 2 million dollars", vec(1.0))])
    db.add(DocumentShare(document_id=doc.id, user_id=bob.id))
    db.commit()

    assert hybrid_search(db, "falcon budget", user_id=bob.id)
    assert hybrid_search(db, "falcon budget", user_id=carol.id) == []


def test_company_documents_are_visible_to_everyone(db):
    mia = make_user(db, "mia@example.com", "manager")
    bob = make_user(db, "bob@example.com")
    add_document(db, mia, "policy.txt", "company", [("Business class needs approval", vec(1.0))])

    assert hybrid_search(db, "business class", user_id=bob.id)


def test_access_filter_is_applied_before_the_result_limit(db):
    """25 closer private chunks must not push the only visible chunk out of Bob's results."""
    alice = make_user(db, "alice@example.com")
    bob = make_user(db, "bob@example.com")
    add_document(db, alice, "secret.txt", "private", [(f"secret note {i}", vec(1.0)) for i in range(25)])
    company = add_document(db, alice, "policy.txt", "company", [("travel policy text", vec(0.1, 1.0))])

    results = hybrid_search(db, "unrelated words", user_id=bob.id, candidates=20)
    assert {r["document_id"] for r in results} == {company.id}
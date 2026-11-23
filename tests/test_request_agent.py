from app.generation import NOT_FOUND
from app.request_agent import lookup_policy, write_draft


def test_lookup_policy_returns_the_passages_from_the_search_tool(monkeypatch):
    calls = []

    def fake_call_tool(name, arguments, **kwargs):
        calls.append((name, arguments))
        return {
            "results": [
                {
                    "chunk_id": 1,
                    "document_id": 1,
                    "filename": "policy.txt",
                    "page_number": 1,
                    "content": "Business class needs approval.",
                }
            ]
        }

    monkeypatch.setattr("app.request_agent.call_tool", fake_call_tool)
    update = lookup_policy({"request_text": "book business class", "requester_id": 7})

    assert len(update["passages"]) == 1
    assert calls == [
        ("search_documents", {"query": "book business class", "user_id": 7, "top_k": 4})
    ]


def test_no_passages_means_not_found_without_calling_the_model(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("the model must not be called")

    monkeypatch.setattr("app.request_agent.generate_answer", must_not_run)
    update = write_draft(
        {"request_text": "buy a laptop", "passages": [], "requester_id": 7}
    )
    assert update["draft"] == NOT_FOUND
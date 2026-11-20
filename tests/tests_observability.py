import json
import logging

from app.observability import JsonFormatter, request_id_var


def test_request_id_is_generated_and_echoed(client):
    generated = client.get("/health").headers["x-request-id"]
    assert len(generated) >= 8
    echoed = client.get("/health", headers={"X-Request-ID": "eval-a01-ask"})
    assert echoed.headers["x-request-id"] == "eval-a01-ask"


def test_unsafe_request_ids_are_cleaned(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id;<script>"})
    assert r.headers["x-request-id"] == "badidscript"


def test_json_log_line_carries_the_request_id():
    token = request_id_var.set("req-123")
    try:
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", None, None)
        record.fields = {"status": 200}
        payload = json.loads(JsonFormatter().format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello"
    assert payload["request_id"] == "req-123"
    assert payload["status"] == 200
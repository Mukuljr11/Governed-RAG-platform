import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
BASE_URL = "http://localhost:8000"
EVAL_EMAIL = "evalbot@example.com"
EVAL_PASSWORD = "evalbot-pass-123"
REFUSAL_MARKER = "couldn't find"

def wait_for_api(timeout: int = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{BASE_URL}/health", timeout=5).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(3)
    raise SystemExit("API did not become ready within 3 minutes")


def make_client() -> httpx.Client:
    client = httpx.Client(
        base_url=BASE_URL,
        timeout=900,
        limits=httpx.Limits(max_keepalive_connections=0),  # new connection per request
    )
    client.post("/auth/register", json={"email": EVAL_EMAIL, "password": EVAL_PASSWORD})
    login = client.post("/auth/login", data={"username": EVAL_EMAIL, "password": EVAL_PASSWORD})
    login.raise_for_status()
    client.headers["Authorization"] = f"Bearer {login.json()['access_token']}"
    return client


def ensure_corpus(client: httpx.Client) -> None:
    existing = {d["filename"] for d in client.get("/documents").json()}
    for path in sorted((ROOT / "corpus").glob("*.txt")):
        if path.name in existing:
            continue
        resp = client.post(
            "/documents/upload",
            files={"file": (path.name, path.read_bytes(), "text/plain")},
            data={"visibility": "private"},
        )
        resp.raise_for_status()
        doc_id = resp.json()["document_id"]
        status: dict = {}
        for _ in range(150):
            status = client.get(f"/documents/{doc_id}/status").json()
            if status["status"] in ("done", "failed"):
                break
            time.sleep(2)
        if status.get("status") != "done":
            raise SystemExit(f"Ingestion of {path.name} did not finish: {status}")
        print(f"ingested {path.name}")


def contains_all(text: str, parts: list[str]) -> bool:
    return all(p.lower() in text for p in parts)


def contains_any(text: str, parts: list[str]) -> bool:
    return not parts or any(p.lower() in text for p in parts)


def retrieval_rank(client: httpx.Client, item: dict):
    resp = client.post("/search", json={"query": item["question"], "top_k": 5})
    if resp.status_code != 200:
        return None
    keyword = item["snippet_keyword"].lower()
    for rank, result in enumerate(resp.json()["results"], start=1):
        if keyword in result["content"].lower():
            return rank
    return None


def evaluate(client: httpx.Client, item: dict, mode: str) -> dict:
    row = {"id": item["id"], "type": item["type"]}
    kind = item["type"]
    start = time.time()

    if kind == "injection":
        resp = client.post("/ask", json={"question": item["question"]})
        row["latency"] = time.time() - start
        row["passed"] = resp.status_code == 400
        return row

    endpoint = "/ask/agent" if mode == "agent" else "/ask"
    resp = client.post(endpoint, json={"question": item["question"]})
    row["latency"] = time.time() - start
    if resp.status_code != 200:
        row.update(passed=False, error=resp.status_code)
        return row

    data = resp.json()
    answer = data["answer"].lower()
    refused = REFUSAL_MARKER in answer
    row["grounded"] = bool(data["grounded"])

    if kind == "unanswerable":
        row["passed"] = refused and not data["sources"]
    elif kind == "pii":
        row["passed"] = not any(p in json.dumps(data) for p in item["pii"])
    else:
        answer_ok = (
            not refused
            and contains_all(answer, item.get("expected_all", []))
            and contains_any(answer, item.get("expected_any", []))
        )
        citation_ok = any(
            item["snippet_keyword"].lower() in s["snippet"].lower() for s in data["sources"]
        )
        row.update(
            answer_ok=answer_ok,
            citation_ok=citation_ok,
            passed=answer_ok and citation_ok and row["grounded"],
            retrieval_rank=retrieval_rank(client, item),
        )
    return row


def mean(values: list[float]):
    return round(sum(values) / len(values), 3) if values else None


def percentile(values: list[float], p: int):
    if not values:
        return None
    values = sorted(values)
    index = max(0, min(len(values) - 1, round(p / 100 * (len(values) - 1))))
    return round(values[index], 1)


def rate(rows: list[dict], key: str = "passed"):
    return mean([1.0 if r.get(key) else 0.0 for r in rows])


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["type"] in ("answerable", "paraphrase")]
    ranks = [r.get("retrieval_rank") for r in answerable if "retrieval_rank" in r]
    latencies = [r["latency"] for r in rows if r["type"] != "injection"]
    return {
        "answer_accuracy": rate(answerable),
        "citation_accuracy": rate(answerable, "citation_ok"),
        "grounded_rate": rate(answerable, "grounded"),
        "retrieval_hit@1": mean([1.0 if rk == 1 else 0.0 for rk in ranks]),
        "retrieval_hit@3": mean([1.0 if rk and rk <= 3 else 0.0 for rk in ranks]),
        "retrieval_mrr": mean([1.0 / rk if rk else 0.0 for rk in ranks]),
        "correct_refusal_rate": rate([r for r in rows if r["type"] == "unanswerable"]),
        "injection_block_rate": rate([r for r in rows if r["type"] == "injection"]),
        "pii_protection_rate": rate([r for r in rows if r["type"] == "pii"]),
        "latency_p50": percentile(latencies, 50),
        "latency_p95": percentile(latencies, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ask", "agent"], default="ask")
    parser.add_argument("--types", nargs="*", help="only run these question types")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--out", help="where to write the results file")
    args = parser.parse_args()

    lines = (ROOT / "golden.jsonl").read_text().splitlines()
    items = [json.loads(line) for line in lines if line.strip()]
    if args.types:
        items = [i for i in items if i["type"] in args.types]
    if args.limit:
        items = items[: args.limit]

    wait_for_api()
    client = make_client()
    ensure_corpus(client)

    rows = []
    for item in items:
        row = evaluate(client, item, args.mode)
        rows.append(row)
        verdict = "PASS" if row.get("passed") else "FAIL"
        print(f"{verdict}  {row['id']}  {row['latency']:.0f}s  {item['question'][:60]}")

    summary = summarize(rows)
    out = Path(args.out) if args.out else (
        ROOT / "results" / f"{args.mode}-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "summary": summary, "rows": rows}, indent=2))

    print(f"\n| metric ({args.mode}) | value |\n|---|---|")
    for key, value in summary.items():
        print(f"| {key} | {value} |")
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
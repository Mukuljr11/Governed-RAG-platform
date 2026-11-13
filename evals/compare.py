import json
import sys

TOLERANCE = 0.05
HIGHER = [
    "answer_accuracy", "citation_accuracy", "grounded_rate", "retrieval_hit@1",
    "retrieval_hit@3", "retrieval_mrr", "correct_refusal_rate", "injection_block_rate",
    "pii_protection_rate",
]
LOWER = ["latency_p50", "latency_p95"]


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)["summary"]


def main(base_path: str, new_path: str) -> None:
    base, new = load(base_path), load(new_path)
    regressions = []
    print("| metric | baseline | new | change |\n|---|---|---|---|")
    for key in HIGHER + LOWER:
        b, n = base.get(key), new.get(key)
        if b is None or n is None:
            continue
        print(f"| {key} | {b} | {n} | {n - b:+.3f} |")
        if key in HIGHER and n < b - TOLERANCE:
            regressions.append(key)
    if regressions:
        print("\nREGRESSION:", ", ".join(regressions))
        sys.exit(1)
    print("\nNo regressions beyond tolerance.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
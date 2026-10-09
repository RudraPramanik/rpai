"""One-off Phase-9 eval runner. Not a CI harness — use for manual checklist fills."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUESTIONS = json.loads((ROOT / "portfolio_questions.json").read_text(encoding="utf-8"))[
    "questions"
]
OUT = ROOT / "_last_run.json"
API = "http://localhost:8000/api/chat"


def ask(query: str) -> tuple[dict | None, str | None]:
    body = json.dumps({"query": query}).encode()
    req = urllib.request.Request(
        API, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode()), None
    except Exception as exc:  # noqa: BLE001 — record any failure for checklist
        return None, str(exc)


def is_insufficient(answer: str) -> bool:
    low = answer.lower()
    needles = (
        "not documented",
        "enough documented",
        "don't have enough",
        "do not have enough",
        "insufficient",
        "i don't have",
        "i do not have",
    )
    return any(n in low for n in needles)


def grade(q: dict, answer: str, sources: list, err: str | None) -> str:
    if err:
        return "fail"
    if not answer.strip():
        return "fail"
    insuf = is_insufficient(answer)
    expected_src = q.get("expected_sources") or []
    src_paths = " ".join(s.get("source", "") for s in sources).lower()

    if q.get("expect_insufficient"):
        if not insuf:
            return "fail"
        if q["id"] == "u01" and re.search(
            r"(salary is|earns|makes)\s*\$?\d", answer, re.I
        ):
            return "fail"
        if q["id"] == "u05" and "quantum" in answer.lower():
            # Allow "does not use quantum" / insufficient; fail if it affirms quantum
            if re.search(r"\b(uses|used|built with)\b.*quantum|quantum.*(uses|used)", answer, re.I):
                return "fail"
        return "pass"

    if expected_src and sources:
        if not all(slug.lower() in src_paths for slug in expected_src):
            return "review"
    if expected_src and not sources:
        # Project JSON alone can answer some questions; flag for human review
        return "review"
    return "pass"


def main() -> None:
    results = []
    for q in QUESTIONS:
        data, err = ask(q["question"])
        answer = (data or {}).get("answer") or ""
        sources = (data or {}).get("sources") or []
        status = grade(q, answer, sources, err)
        row = {
            "id": q["id"],
            "category": q["category"],
            "status": status,
            "expect_insufficient": bool(q.get("expect_insufficient")),
            "insuf_detected": is_insufficient(answer),
            "sources": [s.get("source", "") for s in sources],
            "answer": answer[:400],
            "error": err,
        }
        results.append(row)
        preview = answer.replace("\n", " ")[:100]
        print(f"{q['id']}: {status} | sources={len(sources)} | {preview}")
        time.sleep(0.25)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    summary: dict[str, int] = {}
    for r in results:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    print("SUMMARY", summary)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()

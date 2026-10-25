"""Demo: load the TechNomad knowledge base, create 10 tickets and draft a reply for each.

    python demo/seed.py            # in-process, data in demo/data
    python demo/seed.py --reset    # start from an empty demo database and index
    python demo/seed.py --url http://127.0.0.1:8000   # against a running server

Seven questions are answered by the documents in demo/kb and should be drafted; three are
not covered by them and should be escalated. Needs an LLM key in .env (see .env.example);
without one every ticket is escalated with reason llm_unavailable.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
KB_DIR = DEMO_DIR / "kb"
DATA_DIR = DEMO_DIR / "data"

# (customer name, email, question, expected status)
TICKETS = [
    ("Айгуль Токтосунова", "aigul@example.com",
     "Здравствуйте! Сколько стоит доставка в Ош и сколько дней ждать заказ?", "drafted"),
    ("Бакыт Асанов", "bakyt@example.com",
     "Купил наушники-вкладыши, вскрыл упаковку, но звук не понравился. Могу я их вернуть?",
     "drafted"),
    ("John Miller", "john.miller@example.com",
     "Hi, I live in Bishkek. Can I pay in cash to the courier when my order arrives?", "drafted"),
    ("Нурлан Жумабеков", "nurlan@example.com",
     "Какая гарантия на ноутбук и куда его нести, если он сломался?", "drafted"),
    ("Жыпар Мамытова", "jypar@example.com",
     "Саламатсызбы! Ноутбукка кепилдик канча убакытка берилет?", "drafted"),
    ("Елена Ким", "elena.kim@example.com",
     "Я вернула товар, который оплачивала банковской картой. Когда мне вернут деньги?",
     "drafted"),
    ("Sarah Lee", "sarah.lee@example.com",
     "Does the warranty cover a cracked smartphone screen?", "drafted"),
    ("Азамат Абдыкеров", "azamat@example.com",
     "Есть ли у вас программа лояльности или бонусные баллы за покупки?", "escalated"),
    ("Мээрим Садыкова", "meerim@example.com",
     "Вы продаёте восстановленные iPhone? Какая на них цена?", "escalated"),
    ("Тимур Осмонов", "timur@example.com",
     "Можно сдать мой старый смартфон в зачёт нового (trade-in)? Сколько вы за него дадите?",
     "escalated"),
]


def in_process_client():
    from fastapi.testclient import TestClient

    from support_copilot.api.app import create_app
    from support_copilot.config import CopilotSettings
    from support_copilot.rag import build_rag_service

    # LLM and retrieval settings come from .env; the data stays in demo/data.
    settings = CopilotSettings(data_dir=DATA_DIR)
    if not settings.company_name:
        settings.company_name = "TechNomad"
    return TestClient(create_app(settings, build_rag_service(settings)))


def check(resp, *ok: int) -> dict | list:
    if resp.status_code not in ok:
        sys.exit(f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text}")
    return resp.json()


def shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", help="base URL of a running server (default: in-process)")
    parser.add_argument("--reset", action="store_true", help="delete demo/data first")
    parser.add_argument("--full", action="store_true", help="print every draft in full")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.url:
        import httpx

        client = httpx.Client(base_url=args.url, timeout=120)
    else:
        if args.reset and DATA_DIR.exists():
            shutil.rmtree(DATA_DIR)
        client = in_process_client()

    health = check(client.get("/health"), 200)
    print(f"LLM enabled: {health['llm_enabled']}")
    if not health["llm_enabled"]:
        print("  No LLM key in .env: every ticket will be escalated (llm_unavailable).")

    print("\nKnowledge base:")
    for path in sorted(KB_DIR.glob("*.txt")):
        files = {"file": (path.name, path.read_bytes(), "text/plain")}
        doc = check(client.post("/kb/upload", files=files), 200, 201)
        note = " (already loaded)" if doc["duplicate"] else ""
        print(f"  {doc['filename']:<14} {doc['chunks']} chunks{note}")

    print("\nTickets:")
    rows, mismatches = [], 0
    for name, email, question, expected in TICKETS:
        ticket = check(client.post("/tickets", json={
            "customer_name": name, "customer_email": email, "question": question,
        }), 201)
        result = check(client.post(f"/tickets/{ticket['id']}/draft"), 201)
        status, draft = result["ticket"]["status"], result["draft"]
        mismatches += status != expected
        rows.append((ticket["id"], question, expected, status, draft))

    for ticket_id, question, expected, status, draft in rows:
        mark = "OK " if status == expected else "!! "
        print(f"\n{mark}#{ticket_id} [{status}] {question}")
        if status == "drafted":
            sources = ", ".join(sorted({c["file"] for c in draft["citations"]}))
            body = draft["text"].split("\n\n")[1]  # the answer between greeting and closing
            print(f"    answer:  {body if args.full else shorten(body, 160)}")
            print(f"    sources: {sources}")
        else:
            print(f"    reason:  {draft['escalation_reason']}")
            print(f"    note:    {shorten(draft['text'], 160)}")
        if args.full:
            print("    ---\n    " + draft["text"].replace("\n", "\n    "))
        if status != expected:
            print(f"    expected {expected}")

    drafted = sum(1 for r in rows if r[3] == "drafted")
    print(f"\nSummary: {drafted} drafted, {len(rows) - drafted} escalated; "
          f"{len(rows) - mismatches}/{len(rows)} as expected.")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())

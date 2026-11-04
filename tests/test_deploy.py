"""Public demo features: per-IP rate limit, demo data at startup, OpenAI-compatible LLM."""

import json

from support_copilot.api.ratelimit import RateLimiter
from support_copilot.config import CopilotSettings
from support_copilot.demo_data import load_demo_data
from support_copilot.rag import OpenAICompatibleClient, build_rag_service

from .conftest import create_ticket, upload


def test_limiter_sliding_window():
    now = [0.0]
    limiter = RateLimiter(2, clock=lambda: now[0])
    assert limiter.hit("a") == 0 and limiter.hit("a") == 0
    assert limiter.hit("a") == 3600
    assert limiter.hit("b") == 0  # other IPs keep their own budget
    now[0] = 3600.0
    assert limiter.hit("a") == 0


def test_draft_and_upload_limited_per_ip(make_client):
    client = make_client(rate_limit_per_hour=2)
    ticket = create_ticket(client)
    a = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}
    # Without the header the peer address is the key: uploads share the budget with drafts.
    assert upload(client).status_code == 201
    assert client.post(f"/tickets/{ticket['id']}/draft").status_code == 201
    assert upload(client).status_code == 429
    for _ in range(2):
        assert client.post(f"/tickets/{ticket['id']}/draft", headers=a).status_code == 201
    resp = client.post(f"/tickets/{ticket['id']}/draft", headers=a)
    assert resp.status_code == 429
    assert resp.json()["detail"]["code"] == "rate_limited"
    assert int(resp.headers["Retry-After"]) > 0
    b = {"X-Forwarded-For": "2.2.2.2"}
    assert client.post(f"/tickets/{ticket['id']}/draft", headers=b).status_code == 201
    # Reads and ticket creation are not limited.
    assert client.get("/tickets", headers=a).status_code == 200
    assert client.post("/tickets", json={"customer_name": "A", "customer_email": "a@example.com",
                                         "question": "?"}, headers=a).status_code == 201


def test_no_limit_by_default(client):
    ticket = create_ticket(client)
    for _ in range(5):
        assert client.post(f"/tickets/{ticket['id']}/draft").status_code == 201


def test_demo_data_loaded_once(make_client, tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "delivery.txt").write_text("Доставка в Ош стоит 300 сом.", encoding="utf-8")
    (kb / "returns.txt").write_text("Возврат в течение 14 дней.", encoding="utf-8")
    tickets = tmp_path / "tickets.json"
    tickets.write_text(json.dumps([{"customer_name": "Айгуль", "customer_email":
                                    "aigul@example.com", "question": "Доставка в Ош?"}]),
                       encoding="utf-8")
    client = make_client(demo_kb_dir=kb, demo_tickets_file=tickets)
    state = client.app.state
    load_demo_data(state.settings, state.rag, state.engine)
    load_demo_data(state.settings, state.rag, state.engine)  # restart: not loaded twice

    assert [d["filename"] for d in client.get("/kb").json()] == ["delivery.txt", "returns.txt"]
    assert [t["status"] for t in client.get("/tickets").json()] == ["new"]
    assert client.get("/health").json()["documents"] == 2


def test_llm_base_url(tmp_path):
    settings = CopilotSettings(
        _env_file=None, data_dir=tmp_path, llm_provider="openai", openai_api_key="gsk-test",
        llm_base_url="https://api.groq.com/openai/v1", llm_model="openai/gpt-oss-20b",
    )
    llm = build_rag_service(settings).llm
    assert isinstance(llm, OpenAICompatibleClient)
    assert llm.model == "openai/gpt-oss-20b"
    assert str(llm._client.base_url).rstrip("/") == "https://api.groq.com/openai/v1"


def test_no_base_url_keeps_engine_client(tmp_path):
    settings = CopilotSettings(_env_file=None, data_dir=tmp_path, llm_provider="openai",
                               openai_api_key="sk-test")
    assert not isinstance(build_rag_service(settings).llm, OpenAICompatibleClient)

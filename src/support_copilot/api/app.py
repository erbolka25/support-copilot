"""FastAPI application factory."""

from __future__ import annotations

from document_search.service import RagService
from fastapi import FastAPI
from sqlalchemy import Engine

from support_copilot.api import kb, tickets
from support_copilot.api.deps import RagDep
from support_copilot.config import CopilotSettings
from support_copilot.db import init_db, make_engine
from support_copilot.schemas import Health


def create_app(settings: CopilotSettings, rag: RagService, engine: Engine | None = None) -> FastAPI:
    """Build the app around a ready RagService, so tests can pass fakes."""
    engine = engine or make_engine(settings.resolved_database_url)
    init_db(engine)

    app = FastAPI(
        title="AI Customer Support Copilot",
        version="0.1.0",
        description="Drafts support replies from the knowledge base and escalates the rest.",
    )
    app.state.settings = settings
    app.state.rag = rag
    app.state.engine = engine
    app.include_router(kb.router)
    app.include_router(tickets.router)

    @app.get("/health", response_model=Health, tags=["service"])
    def health(rag: RagDep) -> Health:
        stats = rag.stats()
        return Health(status="ok", documents=stats["documents"], llm_enabled=stats["llm_enabled"])

    return app

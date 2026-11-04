"""The only place that wires document_search into the copilot."""

from __future__ import annotations

from document_search.config import Settings as RagSettings
from document_search.generation import OpenAIClient
from document_search.service import RagService, build_service

from support_copilot.config import CopilotSettings


def rag_settings(settings: CopilotSettings) -> RagSettings:
    """document_search settings built from ours.

    ``_env_file=None`` and an absolute ``data_dir`` keep document_search away from the
    .env and data folder of its own checkout.
    """
    return RagSettings(
        _env_file=None,
        data_dir=settings.rag_data_dir,
        max_upload_mb=settings.max_upload_mb,
        max_documents=settings.max_documents,
        chunk_max_chars=settings.chunk_max_chars,
        chunk_overlap_chars=settings.chunk_overlap_chars,
        index_backend=settings.index_backend,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
        min_score=settings.min_score,
        llm_provider=settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        anthropic_api_key=settings.anthropic_api_key,
        llm_model=settings.llm_model,
        llm_effort=settings.llm_effort,
        llm_max_tokens=settings.llm_max_tokens,
        llm_timeout_s=settings.llm_timeout_s,
        log_level=settings.log_level,
    )


class OpenAICompatibleClient(OpenAIClient):
    """OpenAIClient pointed at another OpenAI-compatible API (Groq, OpenRouter, a local server).

    document_search always talks to api.openai.com, so the bridge swaps in a client with our
    base URL; prompts, error mapping and the citation rules stay the engine's own.
    """

    def __init__(self, base_url: str, api_key: str, **kwargs) -> None:
        super().__init__(api_key=api_key, **kwargs)
        self._client = self._openai.OpenAI(
            api_key=api_key, base_url=base_url, timeout=kwargs.get("timeout_s", 60.0), max_retries=2
        )


def build_rag_service(settings: CopilotSettings) -> RagService:
    """The real service: sentence-transformers embedder (loaded lazily), persisted index, LLM."""
    rs = rag_settings(settings)
    service = build_service(rs)
    api_key = rs.llm_api_key()
    if settings.llm_base_url and settings.llm_provider == "openai" and api_key:
        service.llm = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=api_key,
            model=rs.resolved_llm_model,
            effort=rs.llm_effort,
            max_tokens=rs.llm_max_tokens,
            timeout_s=rs.llm_timeout_s,
        )
    return service

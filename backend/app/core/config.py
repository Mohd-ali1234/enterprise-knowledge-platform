"""Application settings.

Every tunable knob in the platform lives here so that no pipeline stage has to
hard-code a magic number. Values can be overridden through environment
variables or a `.env` file (see `.env.example`).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Unknown variables in .env are ignored rather than fatal, so a stale
        # environment never prevents the service from booting.
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "Enterprise Knowledge Intelligence Platform"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    cors_allow_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── Storage ──────────────────────────────────────────────────
    upload_dir: str = "app/storage/uploads"

    # ── Chunking ─────────────────────────────────────────────────
    chunk_size: int = 600
    chunk_overlap: int = 75

    # ── Archive ingestion (Notion / Confluence exports) ──────────
    archive_max_members: int = 2000
    archive_max_uncompressed_bytes: int = 256 * 1024 * 1024
    """Guards against zip bombs: the sum of member sizes may not exceed this."""

    # ── Web page ingestion ───────────────────────────────────────
    web_fetch_timeout_seconds: float = 30.0
    web_fetch_max_bytes: int = 10 * 1024 * 1024
    web_fetch_user_agent: str = "EKIP-Ingestion/1.0"
    web_fetch_allow_private_hosts: bool = False
    """Off by default: URLs come from clients, and the server would otherwise
    happily fetch localhost and cloud metadata endpoints on their behalf."""

    # ── Knowledge extraction ─────────────────────────────────────
    spacy_model: str = "en_core_web_sm"

    # ── Knowledge graph ──────────────────────────────────────────
    # Extraction is local and deterministic: spaCy dependency parsing, no LLM.
    graph_db_path: str = "./knowledge_graph.db"
    knowledge_graph_enabled: bool = True
    knowledge_graph_relationship_threshold: float = 0.70
    """Relationships scoring below this are discarded. Precision over recall:
    20 accurate edges beat 200 speculative ones."""
    knowledge_graph_entity_merge_threshold: float = 92.0
    """RapidFuzz similarity (0-100) required to treat two names as one entity.
    High on purpose - a wrong merge is far worse than a missed one."""
    knowledge_graph_max_sentence_chars: int = 1000
    knowledge_graph_default_limit: int = 150
    """Nodes returned by an unfiltered graph read, so the UI never tries to
    render an entire large corpus at once."""

    # ── Embeddings ───────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_normalize: bool = True

    # ── Vector store ─────────────────────────────────────────────
    chroma_path: str = "./chroma_db"
    chroma_collection: str = "documents_4"

    # ── Retrieval ────────────────────────────────────────────────
    default_top_k: int = 20
    default_rerank_top_k: int = 5
    rrf_k: int = 60

    # ── Context building ─────────────────────────────────────────
    context_token_budget: int = 2500
    context_chars_per_token: int = 4

    # ── Answer routing ───────────────────────────────────────────
    route_confidence_threshold: float = 0.9
    route_complexity_threshold: float = 0.7

    # ── Gemini (query rewriting + answer generation) ─────────────
    # Without an API key every Gemini-backed stage degrades to its offline
    # counterpart, so the service still runs unconfigured.
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_seconds: float = 60.0
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 1024

    # ── Query rewriting (Ollama) ─────────────────────────────────
    # Retained as an alternative rewriter; not wired in the composition root.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_timeout_seconds: float = 120.0
    ollama_temperature: float = 0.3


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()

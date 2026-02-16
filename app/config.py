"""Centralised configuration loaded from .env via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────
    anthropic_api_key: str

    # ── Neo4j ──────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "changeme"

    # ── Embedding model ────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # ── LLM defaults ──────────────────────────────────────────
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # ── Retrieval defaults ─────────────────────────────────────
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.35
    retrieval_max_hops: int = 3
    rerank_threshold: float = 0.25  # minimum relevance score to keep context elements

    # ── ChromaDB ───────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_data"
    chroma_collection: str = "source_chunks"


settings = Settings()  # type: ignore[call-arg]

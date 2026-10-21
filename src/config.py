"""Centralized configuration for the Healthcare AI Chatbot.

Loads all settings from environment variables with sensible defaults.
No secrets are hardcoded — see .env.example for the full variable list.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    GEMINI = "gemini"
    GROQ = "groq"


def _default_provider() -> LLMProvider:
    """Resolve the active provider.

    Uses LLM_PROVIDER if set; otherwise picks whichever provider has an API key,
    preferring Gemini when both (or neither) are configured.
    """
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return LLMProvider(explicit)
    if not os.getenv("GEMINI_API_KEY") and os.getenv("GROQ_API_KEY"):
        return LLMProvider.GROQ
    return LLMProvider.GEMINI


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for Google Gemini."""

    api_key: str = field(repr=False, default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    )


@dataclass(frozen=True)
class GroqConfig:
    """Configuration for Groq."""

    api_key: str = field(repr=False, default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
    )


@dataclass(frozen=True)
class LLMConfig:
    """Shared LLM generation settings."""

    provider: LLMProvider = field(default_factory=_default_provider)
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.3")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048")))
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    groq: GroqConfig = field(default_factory=GroqConfig)


@dataclass(frozen=True)
class RAGConfig:
    """Configuration for the Retrieval-Augmented Generation pipeline."""

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    hf_token: str = field(repr=False, default_factory=lambda: os.getenv("HF_TOKEN", ""))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "4")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "150")))
    # Hybrid retrieval weights: dense (embedding cosine) vs. lexical (BM25)
    dense_weight: float = field(default_factory=lambda: float(os.getenv("RAG_DENSE_WEIGHT", "0.7")))
    bm25_weight: float = field(default_factory=lambda: float(os.getenv("RAG_BM25_WEIGHT", "0.3")))
    # Scope check: refuse when the best dense cosine similarity is below the floor.
    # Queries that name one of the five diseases use the lower floor.
    scope_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_SCOPE_THRESHOLD", "0.52"))
    )
    scope_threshold_named: float = field(
        default_factory=lambda: float(os.getenv("RAG_SCOPE_THRESHOLD_NAMED", "0.20"))
    )
    knowledge_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "knowledge")


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration."""

    title: str = "🏥 Healthcare AI Assistant"
    icon: str = "🏥"
    description: str = (
        "An AI-powered healthcare information assistant. "
        "Ask questions about symptoms, nutrition, lifestyle, preventive care, and first aid."
    )
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)


# Available Groq models for the UI selector
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# Available Gemini models for the UI selector
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def get_config() -> AppConfig:
    """Return the application configuration.

    Returns:
        AppConfig: Fully initialized configuration loaded from environment variables.

    Raises:
        ValueError: If required configuration for the active provider is missing.
    """
    config = AppConfig()

    # Validate API key for the active provider
    if config.llm.provider == LLMProvider.GEMINI and not config.llm.gemini.api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Please copy .env.example to .env and add your Gemini API key.\n"
            "Get a free key at: https://aistudio.google.com/apikey"
        )

    if config.llm.provider == LLMProvider.GROQ and not config.llm.groq.api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please copy .env.example to .env and add your Groq API key.\n"
            "Get a free key at: https://console.groq.com/keys"
        )

    return config

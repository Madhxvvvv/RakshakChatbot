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
        default_factory=lambda: os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    )


@dataclass(frozen=True)
class LLMConfig:
    """Shared LLM generation settings."""

    provider: LLMProvider = field(
        default_factory=lambda: LLMProvider(os.getenv("LLM_PROVIDER", "gemini").lower())
    )
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
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "3")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "100")))
    knowledge_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data" / "knowledge")
    vectorstore_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "vectorstore")


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
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
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

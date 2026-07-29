"""Embedding model wrapper for the Healthcare AI Chatbot.

Uses LangChain's HuggingFaceEmbeddings for generating text embeddings
in the RAG pipeline. The model is downloaded and run locally — no
HuggingFace API key is required for public models.
"""

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import RAGConfig

logger = logging.getLogger(__name__)


def get_embedding_model(config: RAGConfig) -> HuggingFaceEmbeddings:
    """Create and return a HuggingFace embedding model via LangChain.

    The model is downloaded locally on first use and cached for subsequent calls.
    For the default model (all-MiniLM-L6-v2), no API key is needed.

    Args:
        config: RAG configuration containing the embedding model name and optional HF token.

    Returns:
        A LangChain HuggingFaceEmbeddings instance.
    """
    model_kwargs: dict = {"device": "cpu"}
    encode_kwargs: dict = {"normalize_embeddings": True}

    # Pass HF token if available (for private models or higher rate limits)
    hf_kwargs: dict = {}
    if config.hf_token:
        hf_kwargs["token"] = config.hf_token

    logger.info("Loading embedding model: %s ...", config.embedding_model)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        **hf_kwargs,
    )

    logger.info("Embedding model loaded successfully: %s", config.embedding_model)
    return embeddings

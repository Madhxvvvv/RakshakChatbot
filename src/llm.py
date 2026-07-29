"""LLM client for the Healthcare AI Chatbot.

Provides a factory function to create LangChain chat models for
Gemini and Groq, with a unified interface for both streaming and
non-streaming generation.
"""

import logging
from typing import Generator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


def get_llm(
    config: LLMConfig,
    provider_override: LLMProvider | None = None,
    model_override: str | None = None,
) -> BaseChatModel:
    """Factory function to create a LangChain chat model.

    Returns the appropriate LLM based on the configured provider.
    Supports runtime overrides for provider and model (e.g., from UI toggles).

    Args:
        config: LLM configuration with provider settings.
        provider_override: Optional runtime override for the provider.
        model_override: Optional runtime override for the model name.

    Returns:
        A LangChain BaseChatModel instance (ChatGoogleGenerativeAI or ChatGroq).

    Raises:
        ValueError: If the provider is not supported or API key is missing.
    """
    provider = provider_override or config.provider

    if provider == LLMProvider.GEMINI:
        return _create_gemini_llm(config, model_override)
    elif provider == LLMProvider.GROQ:
        return _create_groq_llm(config, model_override)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_response(
    llm: BaseChatModel,
    messages: list[BaseMessage],
) -> str:
    """Generate a complete response from the LLM.

    Args:
        llm: A LangChain chat model instance.
        messages: List of LangChain message objects.

    Returns:
        The generated response text.

    Raises:
        LLMError: If the API call fails.
    """
    try:
        response = llm.invoke(messages)
        content = response.content

        if content:
            return content

        logger.warning("LLM returned empty response.")
        return _fallback_message()

    except Exception as e:
        logger.error("LLM generation failed: %s", str(e), exc_info=True)
        raise LLMError(f"Failed to generate response: {str(e)}") from e


def generate_response_stream(
    llm: BaseChatModel,
    messages: list[BaseMessage],
) -> Generator[str, None, None]:
    """Generate a streaming response from the LLM.

    Yields response tokens as they arrive for real-time display.

    Args:
        llm: A LangChain chat model instance.
        messages: List of LangChain message objects.

    Yields:
        Text chunks as they are generated.

    Raises:
        LLMError: If the API call fails.
    """
    try:
        for chunk in llm.stream(messages):
            if chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error("LLM streaming failed: %s", str(e), exc_info=True)
        raise LLMError(f"Streaming generation failed: {str(e)}") from e


# -------------------------------------------------------------------------
# Private factory methods
# -------------------------------------------------------------------------


def _create_gemini_llm(config: LLMConfig, model_override: str | None = None) -> BaseChatModel:
    """Create a Google Gemini chat model via LangChain.

    Args:
        config: LLM configuration.
        model_override: Optional model name override.

    Returns:
        ChatGoogleGenerativeAI instance.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = config.gemini.api_key
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    model_name = model_override or config.gemini.model_name

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )

    logger.info("Gemini LLM initialized: model=%s, temp=%.2f", model_name, config.temperature)
    return llm


def _create_groq_llm(config: LLMConfig, model_override: str | None = None) -> BaseChatModel:
    """Create a Groq chat model via LangChain.

    Args:
        config: LLM configuration.
        model_override: Optional model name override.

    Returns:
        ChatGroq instance.
    """
    from langchain_groq import ChatGroq

    api_key = config.groq.api_key
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    model_name = model_override or config.groq.model_name

    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    logger.info("Groq LLM initialized: model=%s, temp=%.2f", model_name, config.temperature)
    return llm


def _fallback_message() -> str:
    """Return a graceful fallback when the LLM returns no content."""
    return (
        "I apologize, but I wasn't able to generate a response at this time. "
        "This might be due to content safety filters. Could you try rephrasing "
        "your question? If the issue persists, please try again in a moment."
    )


class LLMError(Exception):
    """Custom exception for LLM-related errors."""

    pass

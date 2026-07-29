"""Core chatbot orchestrator for the Healthcare AI Chatbot.

Wires together all components — safety guardrails, RAG retriever,
prompt builder, LangChain LLM, and output validation — into a single
cohesive pipeline that processes user queries end-to-end.
"""

import logging
from typing import Generator, Optional

from langchain_core.language_models import BaseChatModel

from src.config import AppConfig, LLMProvider
from src.history import ConversationHistory
from src.llm import LLMError, generate_response, generate_response_stream, get_llm
from src.prompts import (
    DIAGNOSIS_REDIRECT,
    DISCLAIMER_TEXT,
    EMERGENCY_RESPONSE,
    OFF_TOPIC_REDIRECT,
    PRESCRIPTION_REDIRECT,
    build_prompt,
)
from src.safety import SafetyAction, SafetyGuardrails
from src.utils import format_sources, sanitize_input

logger = logging.getLogger(__name__)


class HealthcareChatbot:
    """Main chatbot orchestrator.

    Processes user queries through a layered pipeline:
    1. Input sanitization
    2. Safety guardrails (emergency, diagnosis, prescription, off-topic)
    3. RAG retrieval (if enabled)
    4. Prompt building (system + profile + context + history + query)
    5. LLM generation via LangChain (streaming)
    6. Output validation
    7. Response formatting with sources

    Args:
        config: Application configuration.
        history: Conversation history manager.
        rag_pipeline: Optional RAG pipeline for retrieval-augmented generation.
        provider_override: Optional runtime LLM provider override.
        model_override: Optional runtime model name override.
    """

    def __init__(
        self,
        config: AppConfig,
        history: ConversationHistory,
        rag_pipeline: Optional[object] = None,
        provider_override: LLMProvider | None = None,
        model_override: str | None = None,
    ) -> None:
        self._config = config
        self._history = history
        self._rag = rag_pipeline
        self._safety = SafetyGuardrails()
        self._llm: BaseChatModel = get_llm(
            config.llm,
            provider_override=provider_override,
            model_override=model_override,
        )
        self._provider = provider_override or config.llm.provider
        self._model = model_override
        logger.info(
            "Healthcare Chatbot initialized (provider=%s).",
            self._provider.value,
        )

    def update_llm(
        self,
        provider: LLMProvider,
        model_name: str | None = None,
    ) -> None:
        """Switch to a different LLM provider/model at runtime.

        Args:
            provider: The LLM provider to switch to.
            model_name: Optional specific model name.
        """
        self._llm = get_llm(
            self._config.llm,
            provider_override=provider,
            model_override=model_name,
        )
        self._provider = provider
        self._model = model_name
        logger.info("LLM switched to provider=%s, model=%s", provider.value, model_name)

    def process_query(
        self,
        user_query: str,
        user_health_profile: Optional[dict[str, str]] = None,
    ) -> dict:
        """Process a user query through the full pipeline (non-streaming).

        Args:
            user_query: The raw user input.
            user_health_profile: Optional user health details for personalization.

        Returns:
            Dict with keys:
                - 'response': The final response text.
                - 'sources': Optional list of RAG source citations.
                - 'safety_action': The safety action that was taken.
                - 'is_safety_redirect': Whether the response bypassed the LLM.
        """
        # Step 1: Sanitize input
        query = sanitize_input(user_query)
        if not query:
            return {
                "response": "Please enter a question and I'll be happy to help! 😊",
                "sources": None,
                "safety_action": "proceed",
                "is_safety_redirect": False,
            }

        # Step 2: Safety checks
        safety_result = self._safety.check_input(query)
        logger.info(
            "Safety check: action=%s, reason=%s",
            safety_result.action.value,
            safety_result.reason,
        )

        # Step 3: Handle safety redirects (bypass LLM)
        if safety_result.action != SafetyAction.PROCEED:
            response = self._get_safety_response(safety_result.action)
            return {
                "response": response,
                "sources": None,
                "safety_action": safety_result.action.value,
                "is_safety_redirect": True,
            }

        # Step 4: RAG retrieval
        retrieved_context = None
        sources = None
        if self._rag:
            try:
                retrieved = self._rag.retrieve(query, top_k=self._config.rag.top_k)
                if retrieved:
                    retrieved_context = [
                        {"content": r.content, "source": r.source} for r in retrieved
                    ]
                    sources = [{"source": r.source} for r in retrieved]
                    logger.info("RAG retrieved %d chunks.", len(retrieved))
            except Exception as e:
                logger.warning("RAG retrieval failed, proceeding without context: %s", e)

        # Step 5: Build prompt (returns LangChain messages)
        conversation_history = self._history.get_context_history()
        messages = build_prompt(
            user_query=query,
            conversation_history=conversation_history,
            retrieved_context=retrieved_context,
            user_health_profile=user_health_profile,
        )

        # Step 6: Generate LLM response via LangChain
        try:
            response = generate_response(self._llm, messages)
        except LLMError as e:
            logger.error("LLM error: %s", e)
            return {
                "response": self._error_message(),
                "sources": None,
                "safety_action": "error",
                "is_safety_redirect": False,
            }

        # Step 7: Validate output
        response = self._safety.validate_output(response)

        # Step 8: Append source citations
        if sources:
            response += format_sources(sources)

        return {
            "response": response,
            "sources": sources,
            "safety_action": "proceed",
            "is_safety_redirect": False,
        }

    def process_query_stream(
        self,
        user_query: str,
        user_health_profile: Optional[dict[str, str]] = None,
    ) -> Generator[dict, None, None]:
        """Process a user query with streaming LLM output.

        Yields partial results as the LLM generates tokens. Safety checks
        and RAG retrieval happen before streaming begins.

        Args:
            user_query: The raw user input.
            user_health_profile: Optional user health details.

        Yields:
            Dicts with 'chunk' (text), 'sources', 'done' (bool), and 'is_safety_redirect'.
        """
        # Step 1: Sanitize
        query = sanitize_input(user_query)
        if not query:
            yield {
                "chunk": "Please enter a question and I'll be happy to help! 😊",
                "sources": None,
                "done": True,
                "is_safety_redirect": False,
            }
            return

        # Step 2: Safety checks
        safety_result = self._safety.check_input(query)
        logger.info("Safety check: %s", safety_result.action.value)

        # Step 3: Safety redirect (no streaming needed)
        if safety_result.action != SafetyAction.PROCEED:
            yield {
                "chunk": self._get_safety_response(safety_result.action),
                "sources": None,
                "done": True,
                "is_safety_redirect": True,
            }
            return

        # Step 4: RAG retrieval
        retrieved_context = None
        sources = None
        if self._rag:
            try:
                retrieved = self._rag.retrieve(query, top_k=self._config.rag.top_k)
                if retrieved:
                    retrieved_context = [
                        {"content": r.content, "source": r.source} for r in retrieved
                    ]
                    sources = [{"source": r.source} for r in retrieved]
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)

        # Step 5: Build prompt (LangChain messages)
        conversation_history = self._history.get_context_history()
        messages = build_prompt(
            user_query=query,
            conversation_history=conversation_history,
            retrieved_context=retrieved_context,
            user_health_profile=user_health_profile,
        )

        # Step 6: Stream LLM response via LangChain
        full_response = ""
        try:
            for text_chunk in generate_response_stream(self._llm, messages):
                full_response += text_chunk
                yield {
                    "chunk": text_chunk,
                    "sources": None,
                    "done": False,
                    "is_safety_redirect": False,
                }
        except LLMError as e:
            logger.error("LLM streaming error: %s", e)
            yield {
                "chunk": self._error_message(),
                "sources": None,
                "done": True,
                "is_safety_redirect": False,
            }
            return

        # Step 7: Post-generation validation
        validated = self._safety.validate_output(full_response)
        extra_text = validated[len(full_response):]  # Only the appended parts

        # Step 8: Append any validation additions + sources
        if sources:
            extra_text += format_sources(sources)

        if extra_text:
            yield {
                "chunk": extra_text,
                "sources": sources,
                "done": True,
                "is_safety_redirect": False,
            }
        else:
            yield {
                "chunk": "",
                "sources": sources,
                "done": True,
                "is_safety_redirect": False,
            }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_safety_response(action: SafetyAction) -> str:
        """Get the appropriate safety response for a given action."""
        responses = {
            SafetyAction.EMERGENCY: EMERGENCY_RESPONSE,
            SafetyAction.BLOCK_DIAGNOSIS: DIAGNOSIS_REDIRECT,
            SafetyAction.BLOCK_PRESCRIPTION: PRESCRIPTION_REDIRECT,
            SafetyAction.REDIRECT_OFF_TOPIC: OFF_TOPIC_REDIRECT,
        }
        return responses.get(action, "I'm sorry, I can only help with healthcare-related questions.")

    @staticmethod
    def _error_message() -> str:
        """Return a user-friendly error message."""
        return (
            "I apologize, but I'm experiencing a technical issue and cannot process your "
            "request right now. Please try again in a moment.\n\n"
            "If this persists, it may be due to:\n"
            "- API rate limiting\n"
            "- Temporary service disruption\n"
            "- Content safety filters\n\n"
            "Try rephrasing your question or asking about a different health topic."
        )

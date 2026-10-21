"""Core chatbot orchestrator for the Healthcare AI Chatbot.

Wires together all components — safety guardrails, hybrid retriever with
scope check, prompt builder, LangChain LLM, and output validation — into a
single pipeline that processes user queries end-to-end.
"""

import logging
import re
from dataclasses import dataclass
from typing import Generator, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import AppConfig, LLMProvider
from src.history import ConversationHistory
from src.llm import LLMError, generate_response, generate_response_stream, get_llm
from src.prompts import (
    DIAGNOSIS_REDIRECT,
    EMERGENCY_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    PRESCRIPTION_REDIRECT,
    build_prompt,
)
from src.rag import RAGPipeline, mentions_scoped_disease
from src.safety import SafetyAction, SafetyGuardrails
from src.utils import sanitize_input

logger = logging.getLogger(__name__)

EMPTY_QUERY_RESPONSE = "Please enter a question and I'll be happy to help! 😊"

# Words that refer back to a disease named earlier in the conversation
_FOLLOW_UP_RE = re.compile(
    r"\b(?:it|its|it's|this|that|these|those|they|them|their|the\s+disease|same|yeh|ye|isme|iska|iske|isse)\b",
    re.IGNORECASE,
)


@dataclass
class _Prepared:
    """Result of the pre-generation stages for one query.

    Exactly one of `early_response` (a canned reply that bypasses the LLM)
    or `messages` (the prompt to send to the LLM) is set.
    """

    route: str
    early_response: Optional[str] = None
    messages: Optional[list[BaseMessage]] = None
    sources: Optional[list[dict]] = None


class HealthcareChatbot:
    """Main chatbot orchestrator.

    Processes user queries through a layered pipeline:
    1. Input sanitization
    2. Safety pre-check (emergency, diagnosis, prescription) — no model call
    3. Hybrid retrieval (dense + BM25)
    4. Scope check (similarity floor) — refuse if out of scope
    5. Grounded generation with numbered sources (streaming)
    6. Output validation (dosage warning, disclaimer)

    Args:
        config: Application configuration.
        history: Conversation history manager.
        rag_pipeline: Initialized RAG pipeline over the disease knowledge base.
        provider_override: Optional runtime LLM provider override.
        model_override: Optional runtime model name override.
    """

    def __init__(
        self,
        config: AppConfig,
        history: ConversationHistory,
        rag_pipeline: RAGPipeline,
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

    def process_query(self, user_query: str) -> dict:
        """Process a user query through the full pipeline (non-streaming).

        The conversation history must not yet contain `user_query`.

        Args:
            user_query: The raw user input.

        Returns:
            Dict with keys:
                - 'response': The final response text.
                - 'sources': Optional list of retrieved source dicts.
                - 'route': 'answer', 'out_of_scope', 'error', or a SafetyAction value.
                - 'is_safety_redirect': Whether the response bypassed the LLM.
        """
        prepared = self._prepare(user_query)
        if prepared.early_response is not None:
            return {
                "response": prepared.early_response,
                "sources": None,
                "route": prepared.route,
                "is_safety_redirect": True,
            }

        try:
            response = generate_response(self._llm, prepared.messages)
        except LLMError as e:
            logger.error("LLM error: %s", e)
            return {
                "response": self._error_message(),
                "sources": None,
                "route": "error",
                "is_safety_redirect": False,
            }

        return {
            "response": self._safety.validate_output(response),
            "sources": prepared.sources,
            "route": prepared.route,
            "is_safety_redirect": False,
        }

    def process_query_stream(self, user_query: str) -> Generator[dict, None, None]:
        """Process a user query with streaming LLM output.

        Safety checks, retrieval and the scope check happen before streaming
        begins. The conversation history must not yet contain `user_query`.

        Args:
            user_query: The raw user input.

        Yields:
            Dicts with 'chunk' (text), 'sources', 'done' (bool), 'route' and
            'is_safety_redirect'. After a streamed LLM answer, the final event
            also has 'full_response': the validated text that replaces the
            streamed chunks.
        """
        prepared = self._prepare(user_query)
        if prepared.early_response is not None:
            yield {
                "chunk": prepared.early_response,
                "sources": None,
                "done": True,
                "route": prepared.route,
                "is_safety_redirect": True,
            }
            return

        full_response = ""
        try:
            for text_chunk in generate_response_stream(self._llm, prepared.messages):
                full_response += text_chunk
                yield {
                    "chunk": text_chunk,
                    "sources": None,
                    "done": False,
                    "route": prepared.route,
                    "is_safety_redirect": False,
                }
        except LLMError as e:
            logger.error("LLM streaming error: %s", e)
            yield {
                "chunk": self._error_message(),
                "sources": None,
                "done": True,
                "route": "error",
                "is_safety_redirect": False,
            }
            return

        # Post-generation validation may rewrite citations and append warnings /
        # a disclaimer, so the final event carries the complete validated text.
        yield {
            "chunk": "",
            "full_response": self._safety.validate_output(full_response),
            "sources": prepared.sources,
            "done": True,
            "route": prepared.route,
            "is_safety_redirect": False,
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _prepare(self, user_query: str) -> _Prepared:
        """Run every stage before generation: sanitize, safety, retrieval, scope."""
        query = sanitize_input(user_query)
        if not query:
            return _Prepared(route="empty", early_response=EMPTY_QUERY_RESPONSE)

        # Safety pre-check (deterministic, no model call)
        safety_result = self._safety.check_input(query)
        logger.info(
            "Safety check: action=%s, reason=%s",
            safety_result.action.value,
            safety_result.reason,
        )
        if safety_result.action != SafetyAction.PROCEED:
            return _Prepared(
                route=safety_result.action.value,
                early_response=self._get_safety_response(safety_result.action),
            )

        # Hybrid retrieval + scope check
        conversation_history = self._history.get_context_history()
        retrieval = self._rag.retrieve(
            self._retrieval_query(query, conversation_history),
            top_k=self._config.rag.top_k,
        )
        if not retrieval.in_scope:
            logger.info("Out of scope (top similarity %.3f).", retrieval.top_similarity)
            return _Prepared(route="out_of_scope", early_response=OUT_OF_SCOPE_RESPONSE)

        retrieved_context = [
            {
                "content": c.content,
                "source": c.source,
                "section": c.section,
                "references": c.references,
            }
            for c in retrieval.contexts
        ]
        sources = [
            {
                "source": c.source,
                "section": c.section,
                "references": c.references,
                "similarity": round(c.similarity, 3),
            }
            for c in retrieval.contexts
        ]

        messages = build_prompt(
            user_query=query,
            conversation_history=conversation_history,
            retrieved_context=retrieved_context,
        )
        return _Prepared(route="answer", messages=messages, sources=sources)

    @staticmethod
    def _retrieval_query(query: str, conversation_history: list[dict[str, str]]) -> str:
        """Add the disease from the previous turn to follow-up questions.

        "How does it spread?" after a question about dengue has no disease
        name of its own, so it would retrieve poorly and fail the scope check.
        Only questions that refer back with a pronoun are treated as follow-ups,
        so a new topic ("What are the symptoms of COVID-19?") is still refused.
        """
        if mentions_scoped_disease(query) or not _FOLLOW_UP_RE.search(query):
            return query
        for msg in reversed(conversation_history):
            if msg["role"] == "user" and mentions_scoped_disease(msg["content"]):
                return f"{msg['content']} {query}"
        return query

    @staticmethod
    def _get_safety_response(action: SafetyAction) -> str:
        """Get the appropriate safety response for a given action."""
        responses = {
            SafetyAction.EMERGENCY: EMERGENCY_RESPONSE,
            SafetyAction.BLOCK_DIAGNOSIS: DIAGNOSIS_REDIRECT,
            SafetyAction.BLOCK_PRESCRIPTION: PRESCRIPTION_REDIRECT,
        }
        return responses.get(action, OUT_OF_SCOPE_RESPONSE)

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
            "If you have warning signs such as severe abdominal pain, bleeding, confusion "
            "or difficulty breathing, please seek medical care immediately."
        )

"""Core chatbot orchestrator for the Healthcare AI Chatbot.

Wires together all components — safety guardrails, hybrid retriever with
scope check, prompt builder, LangChain LLM, and output validation — into a
single pipeline that processes user queries end-to-end.

The chatbot holds no conversation state: callers pass the conversation
history with each query, so the same instance can serve the Streamlit app
and the web API.
"""

import logging
import re
from dataclasses import dataclass
from typing import Generator, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from src.config import AppConfig, LLMProvider
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
class Prepared:
    """Result of the pre-generation stages for one query.

    Exactly one of `early_response` (a canned reply that bypasses the LLM)
    or `messages` (the prompt to send to the LLM) is set.

    Attributes:
        route: 'answer', 'out_of_scope', 'empty', or a SafetyAction value.
        early_response: Canned reply, when the LLM is bypassed.
        messages: Prompt for the LLM, when the query should be answered.
        sources: Retrieved source dicts behind the [n] citations.
        top_similarity: Best dense similarity from retrieval (None if not retrieved).
        matched_terms: Safety rule terms that triggered a redirect.
    """

    route: str
    early_response: Optional[str] = None
    messages: Optional[list[BaseMessage]] = None
    sources: Optional[list[dict]] = None
    top_similarity: Optional[float] = None
    matched_terms: Optional[list[str]] = None


class HealthcareChatbot:
    """Main chatbot orchestrator.

    Processes user queries through a layered pipeline:
    1. Input sanitization
    2. Safety pre-check (emergency, diagnosis, prescription) — no model call
    3. Hybrid retrieval (dense + BM25)
    4. Scope check (similarity floor) — refuse if out of scope
    5. Grounded generation with numbered sources
    6. Output validation (citation format, dosage warning, disclaimer)

    Args:
        config: Application configuration.
        rag_pipeline: Initialized RAG pipeline over the disease knowledge base.
        provider_override: Optional runtime LLM provider override.
        model_override: Optional runtime model name override.
        require_llm: Build the LLM. False runs the safety and scope layers only
            (in-scope questions get a setup message instead of an answer).
    """

    def __init__(
        self,
        config: AppConfig,
        rag_pipeline: RAGPipeline,
        provider_override: LLMProvider | None = None,
        model_override: str | None = None,
        require_llm: bool = True,
    ) -> None:
        self._config = config
        self._rag = rag_pipeline
        self._safety = SafetyGuardrails()
        self._llm: Optional[BaseChatModel] = None
        self._provider = provider_override or config.llm.provider
        self._model = model_override
        if require_llm:
            self._llm = get_llm(
                config.llm,
                provider_override=provider_override,
                model_override=model_override,
            )
        logger.info(
            "Healthcare Chatbot initialized (provider=%s, llm=%s).",
            self._provider.value,
            "yes" if self._llm else "no",
        )

    @property
    def has_llm(self) -> bool:
        """Whether a language model is configured (otherwise safety/scope only)."""
        return self._llm is not None

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

    def prepare(
        self,
        user_query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> Prepared:
        """Run every stage before generation: sanitize, safety, retrieval, scope.

        Args:
            user_query: The raw user input.
            conversation_history: Previous messages [{"role", "content"}],
                not including `user_query`.

        Returns:
            A Prepared result: either a canned reply or a prompt for the LLM.
        """
        conversation_history = conversation_history or []
        query = sanitize_input(user_query)
        if not query:
            return Prepared(route="empty", early_response=EMPTY_QUERY_RESPONSE)

        # Safety pre-check (deterministic, no model call)
        safety_result = self._safety.check_input(query)
        logger.info(
            "Safety check: action=%s, reason=%s",
            safety_result.action.value,
            safety_result.reason,
        )
        if safety_result.action != SafetyAction.PROCEED:
            return Prepared(
                route=safety_result.action.value,
                early_response=self._get_safety_response(safety_result.action),
                matched_terms=safety_result.matched_terms,
            )

        # Hybrid retrieval + scope check
        retrieval = self._rag.retrieve(
            self._retrieval_query(query, conversation_history),
            top_k=self._config.rag.top_k,
        )
        if not retrieval.in_scope:
            logger.info("Out of scope (top similarity %.3f).", retrieval.top_similarity)
            return Prepared(
                route="out_of_scope",
                early_response=OUT_OF_SCOPE_RESPONSE,
                top_similarity=retrieval.top_similarity,
            )

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
                "disease": c.disease,
                "section": c.section,
                "references": c.references,
                "retrieved": c.retrieved,
                "similarity": round(c.similarity, 3),
            }
            for c in retrieval.contexts
        ]

        messages = build_prompt(
            user_query=query,
            conversation_history=conversation_history,
            retrieved_context=retrieved_context,
        )
        return Prepared(
            route="answer",
            messages=messages,
            sources=sources,
            top_similarity=retrieval.top_similarity,
        )

    def process_query(
        self,
        user_query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> dict:
        """Process a user query through the full pipeline (non-streaming).

        Args:
            user_query: The raw user input.
            conversation_history: Previous messages, not including `user_query`.

        Returns:
            Dict with keys:
                - 'response': The final response text.
                - 'sources': Optional list of retrieved source dicts.
                - 'route': 'answer', 'out_of_scope', 'setup_required', 'error',
                  'empty', or a SafetyAction value.
                - 'is_safety_redirect': Whether the response bypassed the LLM.
                - 'called_model': Whether the LLM was called.
                - 'top_similarity': Best retrieval similarity, if retrieval ran.
                - 'matched_terms': Safety rule terms, if a rule fired.
        """
        prepared = self.prepare(user_query, conversation_history)
        result = {
            "response": prepared.early_response,
            "sources": None,
            "route": prepared.route,
            "is_safety_redirect": prepared.early_response is not None,
            "called_model": False,
            "top_similarity": prepared.top_similarity,
            "matched_terms": prepared.matched_terms,
        }
        if prepared.early_response is not None:
            return result

        if self._llm is None:
            result.update(response=self._setup_message(), route="setup_required")
            return result

        result["called_model"] = True
        try:
            response = generate_response(self._llm, prepared.messages)
        except LLMError as e:
            logger.error("LLM error: %s", e)
            result.update(response=self._error_message(), route="error")
            return result

        result.update(
            response=self._safety.validate_output(response),
            sources=prepared.sources,
        )
        return result

    def process_query_stream(
        self,
        user_query: str,
        conversation_history: Optional[list[dict[str, str]]] = None,
    ) -> Generator[dict, None, None]:
        """Process a user query with streaming LLM output.

        Safety checks, retrieval and the scope check happen before streaming
        begins.

        Args:
            user_query: The raw user input.
            conversation_history: Previous messages, not including `user_query`.

        Yields:
            Dicts with 'chunk' (text), 'sources', 'done' (bool), 'route' and
            'is_safety_redirect'. After a streamed LLM answer, the final event
            also has 'full_response': the validated text that replaces the
            streamed chunks.
        """
        prepared = self.prepare(user_query, conversation_history)
        if prepared.early_response is not None or self._llm is None:
            yield {
                "chunk": prepared.early_response or self._setup_message(),
                "sources": None,
                "done": True,
                "route": prepared.route if prepared.early_response else "setup_required",
                "is_safety_redirect": prepared.early_response is not None,
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
    def _setup_message() -> str:
        """Reply for an in-scope question when no LLM API key is configured."""
        return (
            "This question is within the knowledge base, but no language model is "
            "configured, so I can't write an answer yet. Add `GROQ_API_KEY` or "
            "`GEMINI_API_KEY` to the `.env` file and restart the server.\n\n"
            "The safety checks and the scope check are still active."
        )

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

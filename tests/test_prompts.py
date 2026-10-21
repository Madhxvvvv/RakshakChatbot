"""Tests for the prompt engineering module.

Covers system prompt content (scope, citation rules), response templates,
and the prompt builder with LangChain message types.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts import (
    DISCLAIMER_TEXT,
    EMERGENCY_RESPONSE,
    OUT_OF_SCOPE_RESPONSE,
    SYSTEM_PROMPT,
    build_prompt,
)


# =============================================================================
# System Prompt Tests
# =============================================================================


class TestSystemPrompt:
    """Tests for the system prompt content."""

    def test_system_prompt_is_not_empty(self) -> None:
        """System prompt should not be empty."""
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_identity(self) -> None:
        """System prompt should define the assistant's identity."""
        assert "health" in SYSTEM_PROMPT.lower()
        assert "assistant" in SYSTEM_PROMPT.lower()

    def test_system_prompt_limits_scope_to_five_diseases(self) -> None:
        """System prompt should name exactly the five in-scope diseases."""
        for disease in ["dengue", "malaria", "typhoid", "tuberculosis", "chikungunya"]:
            assert disease in SYSTEM_PROMPT.lower()

    def test_system_prompt_requires_citations(self) -> None:
        """System prompt should require [n] citations and forbid outside knowledge."""
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "outside knowledge" in SYSTEM_PROMPT.lower()

    def test_out_of_scope_response_lists_diseases(self) -> None:
        """The out-of-scope refusal should tell the user what is in scope."""
        for disease in ["Dengue", "Malaria", "Typhoid", "Tuberculosis", "Chikungunya"]:
            assert disease in OUT_OF_SCOPE_RESPONSE

    def test_system_prompt_forbids_diagnosis(self) -> None:
        """System prompt should explicitly forbid diagnosis."""
        assert "never diagnose" in SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_prescriptions(self) -> None:
        """System prompt should explicitly forbid prescriptions."""
        assert "never prescribe" in SYSTEM_PROMPT.lower()

    def test_system_prompt_encourages_professional_consultation(self) -> None:
        """System prompt should encourage seeing a healthcare professional."""
        assert "healthcare provider" in SYSTEM_PROMPT.lower()

    def test_emergency_response_exists(self) -> None:
        """Emergency response template should exist and contain key info."""
        assert "emergency" in EMERGENCY_RESPONSE.lower()
        assert "911" in EMERGENCY_RESPONSE
        assert "112" in EMERGENCY_RESPONSE
        assert "108" in EMERGENCY_RESPONSE

    def test_disclaimer_exists(self) -> None:
        """Disclaimer text should exist and contain key phrases."""
        assert "not a substitute" in DISCLAIMER_TEXT.lower()
        assert "professional medical advice" in DISCLAIMER_TEXT.lower()


# =============================================================================
# Prompt Builder Tests (LangChain Messages)
# =============================================================================


class TestBuildPrompt:
    """Tests for the build_prompt function with LangChain message types."""

    def test_returns_langchain_messages(self) -> None:
        """build_prompt should return LangChain BaseMessage objects."""
        messages = build_prompt("What is a cold?", [])
        assert all(
            isinstance(m, (SystemMessage, HumanMessage, AIMessage))
            for m in messages
        )

    def test_system_prompt_is_first(self) -> None:
        """System prompt should always be the first message as SystemMessage."""
        messages = build_prompt("What is a cold?", [])
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content.startswith(SYSTEM_PROMPT)

    def test_user_query_is_last(self) -> None:
        """User query should be the last message as HumanMessage."""
        messages = build_prompt("What is a cold?", [])
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "What is a cold?"

    def test_includes_conversation_history(self) -> None:
        """Conversation history should be included with correct message types."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        messages = build_prompt("What is a fever?", history)

        # Find history messages
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage)]

        # At least the history user msg + current query
        assert len(human_msgs) >= 2
        # At least the history assistant msg
        assert len(ai_msgs) >= 1

    def test_includes_numbered_sources(self) -> None:
        """Retrieved chunks should appear in the system message as numbered sources."""
        context = [
            {
                "content": "Aedes mosquitoes bite mainly during the day.",
                "source": "Dengue and severe dengue",
                "section": "Transmission",
                "references": [{"publisher": "WHO", "url": "https://www.who.int"}],
            },
            {"content": "Malaria is spread by Anopheles mosquitoes.", "source": "Malaria"},
        ]
        messages = build_prompt("When do dengue mosquitoes bite?", [], retrieved_context=context)

        system_msg = messages[0].content
        assert "[1] Dengue and severe dengue — Transmission (WHO)" in system_msg
        assert "Aedes mosquitoes bite mainly during the day." in system_msg
        assert "[2] Malaria" in system_msg
        # Sources live in the system message: no extra turns are added
        assert len(messages) == 2

    def test_no_rag_context(self) -> None:
        """Without sources, the model is told it has no verified information."""
        messages = build_prompt("What is a cold?", [])
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)
        assert "no sources were retrieved" in messages[0].content.lower()

    def test_full_pipeline(self) -> None:
        """Test the complete prompt assembly with all components."""
        history = [
            {"role": "user", "content": "Tell me about typhoid"},
            {"role": "assistant", "content": "Typhoid is a bacterial infection [1]."},
        ]
        context = [{"content": "Typhoid spreads through contaminated food.", "source": "Typhoid fever"}]

        messages = build_prompt(
            user_query="How does it spread?",
            conversation_history=history,
            retrieved_context=context,
        )

        assert isinstance(messages[0], SystemMessage)
        assert "contaminated food" in messages[0].content
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "How does it spread?"
        # system + history(2) + user = 4
        assert len(messages) == 4

    def test_message_type_mapping(self) -> None:
        """History roles should map to correct LangChain message types."""
        history = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2"},
        ]
        messages = build_prompt("Q3", history)

        # Skip system message, check history + current query
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]
        expected_types = [HumanMessage, AIMessage, HumanMessage, AIMessage, HumanMessage]
        actual_types = [type(m) for m in non_system]
        assert actual_types == expected_types

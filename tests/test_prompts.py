"""Tests for the prompt engineering module.

Covers system prompt content, prompt builder with LangChain message types,
and health profile injection.
"""

import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.prompts import (
    DISCLAIMER_TEXT,
    EMERGENCY_RESPONSE,
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
        assert "healthcare" in SYSTEM_PROMPT.lower()
        assert "assistant" in SYSTEM_PROMPT.lower()

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
        assert "healthcare" in messages[0].content.lower()

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

    def test_includes_rag_context(self) -> None:
        """RAG context should be injected as HumanMessage + AIMessage pair."""
        context = [
            {"content": "A cold is a viral infection.", "source": "Common Symptoms"},
        ]
        messages = build_prompt("What is a cold?", [], retrieved_context=context)

        # Should have system + RAG human + RAG AI + user query = at least 4 messages
        assert len(messages) >= 4

        # RAG context should appear as a HumanMessage
        rag_found = any(
            isinstance(m, HumanMessage) and "reference information" in m.content.lower()
            for m in messages
        )
        assert rag_found, "RAG context injection not found in messages"

        # Should be followed by an AIMessage acknowledgment
        rag_ack = any(
            isinstance(m, AIMessage) and "evidence-based" in m.content.lower()
            for m in messages
        )
        assert rag_ack, "RAG acknowledgment AIMessage not found"

    def test_no_rag_context(self) -> None:
        """Without RAG context, messages should just be SystemMessage + HumanMessage."""
        messages = build_prompt("What is a cold?", [])
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], HumanMessage)

    def test_health_profile_injection(self) -> None:
        """Health profile should be injected into the SystemMessage content."""
        profile = {
            "Age": "30",
            "Gender": "Male",
            "Known Conditions": "Diabetes",
            "Allergies": "",
        }
        messages = build_prompt("Nutrition tips?", [], user_health_profile=profile)

        system_msg = messages[0].content
        assert "Age: 30" in system_msg
        assert "Gender: Male" in system_msg
        assert "Diabetes" in system_msg
        # Empty fields should not appear
        assert "Allergies" not in system_msg

    def test_empty_health_profile(self) -> None:
        """Empty health profile should not modify system prompt."""
        profile = {"Age": "", "Gender": "", "Known Conditions": ""}
        messages_with = build_prompt("Hi", [], user_health_profile=profile)
        messages_without = build_prompt("Hi", [])

        # System prompts should be identical (no profile injected)
        assert messages_with[0].content == messages_without[0].content

    def test_full_pipeline(self) -> None:
        """Test the complete prompt assembly with all components."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        context = [{"content": "Vitamin C helps immunity.", "source": "Nutrition"}]
        profile = {"Age": "25", "Known Conditions": "Asthma"}

        messages = build_prompt(
            user_query="Tell me about vitamin C",
            conversation_history=history,
            retrieved_context=context,
            user_health_profile=profile,
        )

        # Verify structure
        assert isinstance(messages[0], SystemMessage)
        assert "Asthma" in messages[0].content
        assert isinstance(messages[-1], HumanMessage)
        assert messages[-1].content == "Tell me about vitamin C"
        # system + RAG(2) + history(2) + user = 6
        assert len(messages) >= 6

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

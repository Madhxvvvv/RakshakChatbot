"""Conversation history management for the Healthcare AI Chatbot.

Manages chat messages in Streamlit session_state, providing methods
for adding, retrieving, clearing, and exporting conversation history.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional

import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single message in the conversation.

    Attributes:
        role: The message sender — 'user' or 'assistant'.
        content: The message text content.
        timestamp: ISO-format timestamp of when the message was created.
        sources: Optional list of source citations (from RAG).
    """

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sources: Optional[list[dict]] = None


class ConversationHistory:
    """Manages conversation history using Streamlit session state.

    Stores messages in `st.session_state.messages` as a list of ChatMessage
    objects. Provides methods for adding, retrieving, clearing, and exporting
    the conversation.

    Args:
        max_history: Maximum number of message pairs to retain for context.
                     Older messages are preserved in the display but excluded
                     from the LLM context window to manage token usage.
    """

    SESSION_KEY = "messages"

    def __init__(self, max_history: int = 20) -> None:
        self._max_history = max_history
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Initialize session state if not already done."""
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = []

    @property
    def messages(self) -> list[ChatMessage]:
        """All messages in the conversation."""
        self._ensure_initialized()
        return st.session_state[self.SESSION_KEY]

    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ) -> None:
        """Add a message to the conversation history.

        Args:
            role: 'user' or 'assistant'.
            content: The message text.
            sources: Optional RAG source citations.
        """
        self._ensure_initialized()
        message = ChatMessage(role=role, content=content, sources=sources)
        st.session_state[self.SESSION_KEY].append(message)
        logger.debug("Added %s message (%d chars)", role, len(content))

    def get_context_history(self) -> list[dict[str, str]]:
        """Get recent conversation history formatted for the prompt builder.

        Returns the last `max_history` pairs of messages as simple dicts
        with 'role' and 'content' keys, suitable for passing to
        `prompts.build_prompt()`.

        Returns:
            List of message dicts for prompt context.
        """
        self._ensure_initialized()
        messages = st.session_state[self.SESSION_KEY]

        # Take the most recent messages (up to max_history * 2 for pairs)
        max_messages = self._max_history * 2
        recent = messages[-max_messages:] if len(messages) > max_messages else messages

        return [{"role": msg.role, "content": msg.content} for msg in recent]

    def clear(self) -> None:
        """Clear all conversation history."""
        st.session_state[self.SESSION_KEY] = []
        logger.info("Conversation history cleared.")

    def is_empty(self) -> bool:
        """Check if the conversation history is empty."""
        self._ensure_initialized()
        return len(st.session_state[self.SESSION_KEY]) == 0

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        self._ensure_initialized()
        return len(st.session_state[self.SESSION_KEY])

    def export_to_text(self) -> str:
        """Export the conversation as a formatted text string.

        Returns:
            A human-readable text representation of the conversation.
        """
        self._ensure_initialized()
        if self.is_empty():
            return "No conversation history to export."

        lines = [
            "=" * 60,
            "Public Health Chatbot — Conversation Export",
            f"Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total messages: {self.get_message_count()}",
            "=" * 60,
            "",
        ]

        for msg in st.session_state[self.SESSION_KEY]:
            role_label = "🧑 You" if msg.role == "user" else "🤖 Assistant"
            lines.append(f"{role_label} [{msg.timestamp[:19]}]")
            lines.append(msg.content)
            if msg.sources:
                lines.append("\n📚 Sources:")
                for i, src in enumerate(msg.sources, 1):
                    section = f" — {src['section']}" if src.get("section") else ""
                    urls = ", ".join(r["url"] for r in src.get("references") or [] if r.get("url"))
                    lines.append(f"  [{i}] {src.get('source', 'Unknown')}{section} ({urls})")
            lines.append("")
            lines.append("-" * 40)
            lines.append("")

        return "\n".join(lines)

    def export_to_json(self) -> str:
        """Export the conversation as a JSON string.

        Returns:
            JSON string of all messages.
        """
        self._ensure_initialized()
        messages_data = [asdict(msg) for msg in st.session_state[self.SESSION_KEY]]
        return json.dumps(messages_data, indent=2, ensure_ascii=False)

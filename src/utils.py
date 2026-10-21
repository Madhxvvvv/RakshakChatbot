"""Utility functions for the Healthcare AI Chatbot.

Contains shared helpers for logging setup, text processing,
and other cross-cutting concerns.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure application-wide logging.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def sanitize_input(text: str) -> str:
    """Sanitize user input by stripping excessive whitespace.

    Args:
        text: Raw user input.

    Returns:
        Cleaned input string.
    """
    if not text:
        return ""
    # Collapse multiple spaces/newlines, strip leading/trailing whitespace
    return " ".join(text.split()).strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length with a suffix.

    Args:
        text: Text to truncate.
        max_length: Maximum allowed length.
        suffix: String to append when truncating.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def format_sources(sources: Optional[list[dict]], show_similarity: bool = False) -> str:
    """Format retrieved sources as a numbered markdown list.

    The numbers match the [n] citations in the answer. Each entry names the
    knowledge-base document and section, followed by links to the WHO/CDC
    pages it was written from.

    Args:
        sources: Source dicts with 'source', 'section', 'references'
            ([{publisher, title, url}]) and optionally 'similarity'.
        show_similarity: Append the retrieval similarity (diagnostics).

    Returns:
        Markdown string, or "" if there are no sources.
    """
    if not sources:
        return ""

    lines = []
    for i, src in enumerate(sources, 1):
        label = src.get("source", "Unknown source")
        if src.get("section"):
            label += f" — {src['section']}"
        links = [
            f"[{ref.get('publisher', 'Source')}]({ref['url']})"
            for ref in src.get("references") or []
            if ref.get("url")
        ]
        line = f"**[{i}]** {label}"
        if links:
            line += " · " + " · ".join(links)
        if show_similarity and "similarity" in src:
            line += f" · similarity {src['similarity']:.2f}"
        lines.append(line)

    return "\n\n".join(lines)

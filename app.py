"""Public Health Chatbot for Disease Awareness — Main Application Entry Point.

Run with: streamlit run app.py
"""

import logging

import streamlit as st

from src.chatbot import HealthcareChatbot
from src.config import AppConfig, get_config
from src.history import ConversationHistory
from src.rag import RAGPipeline
from src.ui import (
    display_sources,
    inject_custom_css,
    render_chat_history,
    render_provider_badges,
    render_sidebar,
    render_welcome_message,
)
from src.utils import setup_logging

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Public Health Chatbot",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# AI-Driven Public Health Chatbot for Disease Awareness\n"
            "Answers questions on dengue, malaria, typhoid, tuberculosis and "
            "chikungunya using a curated WHO/CDC knowledge base, with source citations.\n\n"
            "⚠️ This tool is for awareness only. It does not diagnose or prescribe. "
            "Always consult a healthcare professional."
        ),
    },
)


# ---------------------------------------------------------------------------
# Initialization (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading configuration...")
def load_config() -> AppConfig:
    """Load and cache the application configuration."""
    return get_config()


@st.cache_resource(show_spinner="Indexing the disease knowledge base...")
def load_rag_pipeline(_config: AppConfig) -> RAGPipeline | None:
    """Initialize and cache the RAG pipeline.

    Args:
        _config: Application configuration (underscore prefix to skip hashing).

    Returns:
        Initialized RAGPipeline or None if initialization fails.
    """
    try:
        rag = RAGPipeline(_config.rag)
        if rag.initialize():
            logger.info("RAG pipeline initialized successfully.")
            return rag
        logger.error("RAG pipeline initialization failed: no chunks indexed.")
        return None
    except Exception as e:
        logger.error("RAG pipeline error: %s", e, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Main application function."""
    inject_custom_css()

    try:
        config = load_config()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    history = ConversationHistory(max_history=20)

    # The chatbot answers only from the knowledge base, so it cannot run without it.
    rag_pipeline = load_rag_pipeline(config)
    if rag_pipeline is None:
        st.error(
            "⚠️ The disease knowledge base could not be loaded, so the chatbot cannot "
            "answer safely. Check that `data/knowledge/` contains the disease files and "
            "see the terminal logs for details."
        )
        st.stop()

    selected_provider, selected_model, show_diagnostics = render_sidebar(
        history, default_provider=config.llm.provider
    )

    # --- LLM Provider Management ---
    # Track provider/model changes to reinitialize the chatbot
    provider_key = f"{selected_provider.value}:{selected_model}"

    if "current_provider_key" not in st.session_state:
        st.session_state.current_provider_key = None

    needs_reinit = (
        "chatbot" not in st.session_state
        or st.session_state.current_provider_key != provider_key
    )

    if needs_reinit:
        try:
            st.session_state.chatbot = HealthcareChatbot(
                config=config,
                history=history,
                rag_pipeline=rag_pipeline,
                provider_override=selected_provider,
                model_override=selected_model,
            )
            st.session_state.current_provider_key = provider_key
            logger.info(
                "Chatbot initialized: provider=%s, model=%s",
                selected_provider.value,
                selected_model,
            )
        except ValueError as e:
            st.error(
                f"⚠️ Failed to initialize {selected_provider.value.title()} LLM: {str(e)}\n\n"
                f"Please check your API key in the `.env` file."
            )
            st.stop()

    chatbot: HealthcareChatbot = st.session_state.chatbot

    # --- Main Chat Area ---
    st.markdown("# 🦟 Public Health Chatbot")

    render_provider_badges(
        provider=selected_provider,
        model=selected_model,
        rag_enabled=rag_pipeline.is_initialized,
    )

    st.markdown("")  # Spacing

    if history.is_empty():
        render_welcome_message()
    else:
        render_chat_history(history, show_diagnostics=show_diagnostics)

    # --- Handle suggested question ---
    if "suggested_question" in st.session_state:
        user_input = st.session_state.pop("suggested_question")
        _handle_user_input(user_input, chatbot, history, show_diagnostics)

    # --- Chat Input ---
    if user_input := st.chat_input(
        "Ask about dengue, malaria, typhoid, TB or chikungunya...",
        key="chat_input",
    ):
        _handle_user_input(user_input, chatbot, history, show_diagnostics)


def _handle_user_input(
    user_input: str,
    chatbot: HealthcareChatbot,
    history: ConversationHistory,
    show_diagnostics: bool,
) -> None:
    """Process user input and generate a response.

    The user message is added to history only after the response is generated,
    so the chatbot's context history does not already contain the current query.

    Args:
        user_input: The user's message.
        chatbot: The chatbot orchestrator.
        history: Conversation history manager.
        show_diagnostics: Show retrieval similarity next to sources.
    """
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🏥"):
        # Use streaming for real-time token display
        response_placeholder = st.empty()
        full_response = ""
        sources = None

        try:
            for result in chatbot.process_query_stream(user_query=user_input):
                chunk = result.get("chunk", "")
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                if result.get("done"):
                    if result.get("full_response") is not None:
                        full_response = result["full_response"]
                    sources = result.get("sources")
                    break

            # Final render without cursor
            response_placeholder.markdown(full_response)

            if sources:
                display_sources(sources, show_diagnostics=show_diagnostics)

        except Exception as e:
            logger.error("Error processing query: %s", e, exc_info=True)
            full_response = (
                "I apologize, but I encountered an error while processing your question. "
                "Please try again or rephrase your question."
            )
            response_placeholder.markdown(full_response)

    history.add_message("user", user_input)
    history.add_message("assistant", full_response, sources=sources)


if __name__ == "__main__":
    main()

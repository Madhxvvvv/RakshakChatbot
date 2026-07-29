"""Healthcare AI Chatbot — Main Application Entry Point.

Run with: streamlit run app.py
"""

import logging

import streamlit as st

from src.chatbot import HealthcareChatbot
from src.config import AppConfig, LLMProvider, get_config
from src.history import ConversationHistory
from src.rag import RAGPipeline
from src.ui import (
    display_sources,
    inject_custom_css,
    render_chat_history,
    render_provider_badges,
    render_sidebar,
    render_welcome_message,
    render_document_upload,
)
from src.utils import setup_logging
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare AI Assistant",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# Healthcare AI Assistant\n"
            "An AI-powered health information chatbot built with "
            "Streamlit, LangChain, and powered by Gemini / Groq.\n\n"
            "⚠️ This tool is for educational purposes only. "
            "Always consult a healthcare professional for medical advice."
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


@st.cache_resource(show_spinner="Initializing knowledge base (this may take a moment)...")
def load_rag_pipeline_v2(_config: AppConfig) -> RAGPipeline | None:
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
        else:
            logger.warning("RAG pipeline initialization failed. Running without RAG.")
            return None
    except Exception as e:
        logger.error("RAG pipeline error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


def main() -> None:
    """Main application function."""
    # Inject custom styles
    inject_custom_css()

    # Load configuration
    try:
        config = load_config()
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # Initialize conversation history
    history = ConversationHistory(max_history=20)

    # Initialize RAG pipeline (cached)
    rag_pipeline = load_rag_pipeline_v2(config)
    show_uploader = bool(rag_pipeline and rag_pipeline.is_initialized)

    # Render sidebar and get user selections
    health_profile, selected_provider, selected_model, uploaded_file = render_sidebar(
        history, show_uploader=show_uploader
    )

    # --- Document Upload Processing ---
    if show_uploader and uploaded_file is not None:
        # Use file ID (or name + size) to track if it's already ingested
        file_hash = f"{uploaded_file.name}_{uploaded_file.size}"
        if "ingested_files" not in st.session_state:
            st.session_state.ingested_files = set()
            
        if file_hash not in st.session_state.ingested_files:
            with st.spinner("Processing PDF..."):
                # Save to a temporary file that automatically deletes when closed
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = Path(tmp.name)
                    
                # Ingest the temporary file
                chunks_added = rag_pipeline.ingest_single_document(tmp_path)
                
                # Clean up the temporary file manually just to be safe
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Failed to delete temp file {tmp_path}: {e}")
                    
                if chunks_added > 0:
                    st.session_state.ingested_files.add(file_hash)
                    st.sidebar.success(f"Added '{uploaded_file.name}' for this session!")
                else:
                    st.sidebar.error("Failed to process the PDF.")

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
    st.markdown("# 🏥 Healthcare AI Assistant")

    # Show provider and feature badges
    render_provider_badges(
        provider=selected_provider,
        model=selected_model,
        rag_enabled=rag_pipeline is not None and rag_pipeline.is_initialized,
    )

    st.markdown("")  # Spacing

    # Show welcome message or chat history
    if history.is_empty():
        render_welcome_message()
    else:
        render_chat_history(history)

    # --- Handle suggested question ---
    if "suggested_question" in st.session_state:
        user_input = st.session_state.pop("suggested_question")
        _handle_user_input(user_input, chatbot, history, health_profile)

    # --- Chat Input ---
    if user_input := st.chat_input(
        "Ask a health question...",
        key="chat_input",
    ):
        _handle_user_input(user_input, chatbot, history, health_profile)


def _handle_user_input(
    user_input: str,
    chatbot: HealthcareChatbot,
    history: ConversationHistory,
    health_profile: dict[str, str],
) -> None:
    """Process user input and generate a response.

    Args:
        user_input: The user's message.
        chatbot: The chatbot orchestrator.
        history: Conversation history manager.
        health_profile: User's health profile data.
    """
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # Add to history
    history.add_message("user", user_input)

    # Generate and display assistant response
    with st.chat_message("assistant", avatar="🏥"):
        # Use streaming for real-time token display
        response_placeholder = st.empty()
        full_response = ""
        sources = None

        try:
            for result in chatbot.process_query_stream(
                user_query=user_input,
                user_health_profile=health_profile,
            ):
                if result.get("is_safety_redirect") and result.get("chunk"):
                    # Safety redirects are displayed all at once
                    full_response = result["chunk"]
                    response_placeholder.markdown(full_response)
                    sources = result.get("sources")
                    break

                chunk = result.get("chunk", "")
                if chunk:
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                if result.get("done"):
                    sources = result.get("sources")
                    break

            # Final render without cursor
            response_placeholder.markdown(full_response)

            # Display sources if available
            if sources:
                display_sources(sources)

        except Exception as e:
            logger.error("Error processing query: %s", e, exc_info=True)
            full_response = (
                "I apologize, but I encountered an error while processing your question. "
                "Please try again or rephrase your question."
            )
            response_placeholder.markdown(full_response)

    # Add assistant response to history
    history.add_message("assistant", full_response, sources=sources)


if __name__ == "__main__":
    main()

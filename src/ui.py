"""UI helper functions for the Healthcare AI Chatbot.

Contains Streamlit UI components: sidebar with LLM provider selector,
chat rendering, source display, and custom CSS styling.
"""

import streamlit as st

from src.config import GEMINI_MODELS, GROQ_MODELS, LLMProvider
from src.history import ConversationHistory
from src.utils import format_sources


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
<style>
/* --- Global Styles --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.stApp {
    font-family: 'Inter', sans-serif;
}

/* --- Chat Message Styling --- */
.stChatMessage {
    border-radius: 12px;
    margin-bottom: 8px;
}

/* --- Sidebar Styling --- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}

[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

/* --- Disclaimer Banner --- */
.disclaimer-banner {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin-bottom: 16px;
    font-size: 0.85rem;
    color: #92400e;
}

/* --- Emergency Alert --- */
.emergency-alert {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border-left: 4px solid #ef4444;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    margin: 8px 0;
    color: #991b1b;
}

/* --- Source Citation --- */
.source-citation {
    background: #f0f9ff;
    border-left: 3px solid #0ea5e9;
    padding: 8px 12px;
    border-radius: 0 6px 6px 0;
    margin-top: 8px;
    font-size: 0.85rem;
    color: #0c4a6e;
}

/* --- Welcome Card --- */
.welcome-card {
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    border: 1px solid #bfdbfe;
    text-align: center;
}

/* --- Feature Badge --- */
.feature-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

/* --- Provider Badge --- */
.provider-badge-gemini {
    display: inline-block;
    background: linear-gradient(135deg, #1a73e8 0%, #4285f4 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

.provider-badge-groq {
    display: inline-block;
    background: linear-gradient(135deg, #f55036 0%, #ff6b4a 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

/* --- Streamlit Overrides --- */
.stTextInput > div > div > input {
    border-radius: 8px;
}

.stSelectbox > div > div {
    border-radius: 8px;
}

.stTextArea > div > div > textarea {
    border-radius: 8px;
}

/* Smooth transitions */
* {
    transition: background-color 0.2s ease, border-color 0.2s ease;
}

/* --- Footer --- */
.app-footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.75rem;
    padding: 16px;
    margin-top: 32px;
    border-top: 1px solid #e2e8f0;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject custom CSS into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Sidebar Components
# =============================================================================


def render_sidebar(
    history: ConversationHistory,
    default_provider: LLMProvider = LLMProvider.GEMINI,
    show_diagnostics_default: bool = False,
) -> tuple[LLMProvider, str, bool]:
    """Render the sidebar with scope info, LLM selector, and controls.

    No personal or health data is collected (an explicit non-goal of the project).

    Args:
        history: The conversation history manager.
        default_provider: Provider preselected in the dropdown.
        show_diagnostics_default: Initial state of the retrieval diagnostics toggle.

    Returns:
        Tuple of (selected_provider, selected_model, show_diagnostics).
    """
    with st.sidebar:
        # App title and description
        st.markdown("## 🦟 Public Health Chatbot")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.85rem;'>"
            "Disease awareness for <strong>dengue, malaria, typhoid, tuberculosis "
            "and chikungunya</strong>. Every answer is grounded in WHO and CDC guidance "
            "and cites its sources."
            "</p>",
            unsafe_allow_html=True,
        )

        # --- Medical Disclaimer ---
        st.markdown("### ⚠️ Medical Disclaimer")
        st.markdown(
            '<div class="disclaimer-banner">'
            "<strong>Important:</strong> This is an awareness tool only. It does "
            "<strong>not</strong> diagnose, prescribe, or replace a doctor. "
            "In an emergency, call <strong>112</strong> or <strong>108</strong> (India)."
            "</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # --- LLM Provider Selection ---
        st.markdown("### 🤖 LLM Provider")

        provider_options = ["Gemini", "Groq"]
        selected_provider_name = st.selectbox(
            "Select Provider",
            options=provider_options,
            index=0 if default_provider == LLMProvider.GEMINI else 1,
            key="llm_provider_select",
            help="Choose which AI model to use for generating responses.",
        )

        selected_provider = (
            LLMProvider.GEMINI if selected_provider_name == "Gemini" else LLMProvider.GROQ
        )

        # Model selection based on provider
        if selected_provider == LLMProvider.GEMINI:
            model_options = GEMINI_MODELS
            default_model_index = 0
        else:
            model_options = GROQ_MODELS
            default_model_index = 0

        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=default_model_index,
            key="model_select",
            help="Choose the specific model to use.",
        )

        show_diagnostics = st.toggle(
            "🔎 Show retrieval diagnostics",
            value=show_diagnostics_default,
            key="show_diagnostics",
            help="Show the similarity score of each retrieved source.",
        )

        st.divider()

        # --- Controls Section ---
        st.markdown("### ⚙️ Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="btn_clear"):
                history.clear()
                st.rerun()

        with col2:
            if not history.is_empty():
                chat_export = history.export_to_text()
                st.download_button(
                    "📥 Export",
                    data=chat_export,
                    file_name="healthcare_chat_export.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="btn_export",
                )

        # --- App Info ---
        st.divider()
        provider_info = (
            f"Provider: {selected_provider_name}<br>"
            f"Model: {selected_model}<br>"
        )
        st.markdown(
            f"<p style='color: #64748b; font-size: 0.7rem; text-align: center;'>"
            f"{provider_info}"
            f"Built with Streamlit + LangChain<br>"
            f"Hybrid retrieval: embeddings + BM25"
            f"</p>",
            unsafe_allow_html=True,
        )

    return selected_provider, selected_model, show_diagnostics


# =============================================================================
# Chat Display Components
# =============================================================================


def render_welcome_message() -> None:
    """Render the welcome message when chat is empty."""
    st.markdown(
        '<div class="welcome-card">'
        '<h2 style="margin: 0 0 8px 0; color: #1e40af;">👋 Welcome!</h2>'
        '<p style="color: #3b82f6; margin: 0 0 16px 0; font-size: 1.05rem;">'
        "Ask me about dengue, malaria, typhoid, tuberculosis or chikungunya."
        "</p>"
        '<p style="color: #64748b; margin: 0 0 16px 0; font-size: 0.9rem;">'
        "Symptoms, how they spread, prevention, warning signs and when to seek care — "
        "every answer cites WHO and CDC guidance. I can't diagnose or recommend medicines."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick topic suggestions
    st.markdown("#### 💡 Try asking about:")
    cols = st.columns(3)
    sample_questions = [
        ("🦟", "What are the warning signs of severe dengue?"),
        ("🌙", "Do dengue mosquitoes bite only at night?"),
        ("🦟", "How does malaria spread?"),
        ("💧", "How can I prevent typhoid?"),
        ("🫁", "Can TB spread through shared utensils?"),
        ("🦵", "How is chikungunya different from dengue?"),
    ]

    for i, (emoji, question) in enumerate(sample_questions):
        with cols[i % 3]:
            if st.button(
                f"{emoji} {question}",
                key=f"suggestion_{i}",
                use_container_width=True,
            ):
                st.session_state.suggested_question = question
                st.rerun()


def render_chat_history(history: ConversationHistory, show_diagnostics: bool = False) -> None:
    """Render the chat history with styled messages.

    Args:
        history: The conversation history manager.
        show_diagnostics: Show retrieval similarity next to each source.
    """
    for msg in history.messages:
        with st.chat_message(msg.role, avatar="🧑‍💻" if msg.role == "user" else "🏥"):
            st.markdown(msg.content)

            # Show sources if available
            if msg.sources:
                display_sources(msg.sources, show_diagnostics=show_diagnostics)


def render_provider_badges(
    provider: LLMProvider,
    model: str,
    rag_enabled: bool,
) -> None:
    """Render feature and provider badges in the main area.

    Args:
        provider: Active LLM provider.
        model: Active model name.
        rag_enabled: Whether RAG is active.
    """
    badge_class = (
        "provider-badge-gemini" if provider == LLMProvider.GEMINI else "provider-badge-groq"
    )
    provider_label = f"{'🔷 Gemini' if provider == LLMProvider.GEMINI else '🟠 Groq'}: {model}"

    badges = f'<span class="{badge_class}">{provider_label}</span> '

    if rag_enabled:
        badges += '<span class="feature-badge">📚 WHO / CDC sources</span> '

    badges += (
        '<span class="feature-badge">🛡️ Safety Guardrails</span> '
        '<span class="feature-badge">🎯 5 diseases only</span>'
    )

    st.markdown(badges, unsafe_allow_html=True)


def display_sources(sources: list[dict], show_diagnostics: bool = False) -> None:
    """Display the numbered sources behind an answer's [n] citations.

    Args:
        sources: Source dicts with 'source', 'section', 'references', 'similarity'.
        show_diagnostics: Show retrieval similarity for each source.
    """
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        st.markdown(format_sources(sources, show_similarity=show_diagnostics))

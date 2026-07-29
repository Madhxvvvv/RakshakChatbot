"""UI helper functions for the Healthcare AI Chatbot.

Contains Streamlit UI components: sidebar with LLM provider selector,
health profile form, chat rendering, source display, and custom CSS styling.
"""

import streamlit as st

from src.config import GEMINI_MODELS, GROQ_MODELS, LLMProvider
from src.history import ConversationHistory


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

/* --- Health Profile Card --- */
.profile-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    border: 1px solid #475569;
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
    history: ConversationHistory, show_uploader: bool = False
) -> tuple[dict[str, str], LLMProvider, str, object | None]:
    """Render the sidebar with LLM selector, health profile form, and controls.

    Args:
        history: The conversation history manager.
        show_uploader: Whether to show the document uploader.

    Returns:
        Tuple of (health_profile_dict, selected_provider, selected_model, uploaded_file).
    """
    with st.sidebar:
        # App title and description
        st.markdown("## 🏥 Healthcare AI Assistant")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.85rem;'>"
            "Your AI-powered health information companion. "
            "Ask questions about symptoms, nutrition, lifestyle, and more."
            "</p>",
            unsafe_allow_html=True,
        )

        # --- Medical Disclaimer ---
        st.markdown("### ⚠️ Medical Disclaimer")
        st.markdown(
            '<div class="disclaimer-banner">'
            "<strong>Important:</strong> This AI assistant provides general health "
            "information for educational purposes only. It is <strong>not</strong> a "
            "substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult a qualified healthcare provider."
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
            index=0,
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

        st.divider()

        # --- Health Profile Section ---
        st.markdown("### 👤 Your Health Profile")
        st.markdown(
            "<p style='color: #94a3b8; font-size: 0.8rem;'>"
            "Share details for more personalized responses (optional)."
            "</p>",
            unsafe_allow_html=True,
        )

        # Initialize health profile in session state
        if "health_profile" not in st.session_state:
            st.session_state.health_profile = {
                "Age": "",
                "Gender": "",
                "Known Conditions": "",
                "Allergies": "",
                "Current Medications": "",
                "Lifestyle Notes": "",
            }

        profile = st.session_state.health_profile

        col1, col2 = st.columns(2)
        with col1:
            profile["Age"] = st.text_input(
                "Age",
                value=profile.get("Age", ""),
                placeholder="e.g., 30",
                key="profile_age",
            )
        with col2:
            profile["Gender"] = st.selectbox(
                "Gender",
                options=["", "Male", "Female", "Non-binary", "Prefer not to say"],
                index=0,
                key="profile_gender",
            )

        profile["Known Conditions"] = st.text_input(
            "Known Conditions",
            value=profile.get("Known Conditions", ""),
            placeholder="e.g., Diabetes, Hypertension",
            key="profile_conditions",
        )

        profile["Allergies"] = st.text_input(
            "Allergies",
            value=profile.get("Allergies", ""),
            placeholder="e.g., Penicillin, Peanuts",
            key="profile_allergies",
        )

        profile["Current Medications"] = st.text_input(
            "Current Medications",
            value=profile.get("Current Medications", ""),
            placeholder="e.g., Metformin, Lisinopril",
            key="profile_medications",
        )

        profile["Lifestyle Notes"] = st.text_area(
            "Lifestyle Notes",
            value=profile.get("Lifestyle Notes", ""),
            placeholder="e.g., Vegetarian, exercises 3x/week",
            height=68,
            key="profile_lifestyle",
        )

        st.session_state.health_profile = profile

        st.divider()

        # --- Document Uploader ---
        uploaded_file = None
        if show_uploader:
            uploaded_file = render_document_upload()
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
            f"RAG-enabled with FAISS"
            f"</p>",
            unsafe_allow_html=True,
        )

    return profile, selected_provider, selected_model, uploaded_file


# =============================================================================
# Chat Display Components
# =============================================================================


def render_welcome_message() -> None:
    """Render the welcome message when chat is empty."""
    st.markdown(
        '<div class="welcome-card">'
        '<h2 style="margin: 0 0 8px 0; color: #1e40af;">👋 Welcome!</h2>'
        '<p style="color: #3b82f6; margin: 0 0 16px 0; font-size: 1.05rem;">'
        "I'm your Healthcare AI Assistant, here to help with general health questions."
        "</p>"
        '<p style="color: #64748b; margin: 0 0 16px 0; font-size: 0.9rem;">'
        "Ask me about symptoms, nutrition, healthy lifestyle, preventive care, first aid, and more."
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick topic suggestions
    st.markdown("#### 💡 Try asking about:")
    cols = st.columns(3)
    sample_questions = [
        ("🤒", "Common cold symptoms"),
        ("🥗", "Balanced diet tips"),
        ("💪", "Exercise recommendations"),
        ("🩹", "Basic first aid for burns"),
        ("😴", "Better sleep habits"),
        ("🧠", "Managing stress"),
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


def render_chat_history(history: ConversationHistory) -> None:
    """Render the chat history with styled messages.

    Args:
        history: The conversation history manager.
    """
    for msg in history.messages:
        with st.chat_message(msg.role, avatar="🧑‍💻" if msg.role == "user" else "🏥"):
            st.markdown(msg.content)

            # Show sources if available
            if msg.sources:
                display_sources(msg.sources)


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
        badges += '<span class="feature-badge">📚 RAG Enabled</span> '

    badges += (
        '<span class="feature-badge">🛡️ Safety Guardrails</span> '
        '<span class="feature-badge">💬 Conversation Memory</span>'
    )

    st.markdown(badges, unsafe_allow_html=True)


def display_sources(sources: list[dict[str, str]]) -> None:
    """Display RAG source citations in an expander.

    Args:
        sources: List of source dicts with 'source' key.
    """
    if not sources:
        return

    with st.expander("📚 Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            source_name = src.get("source", "Unknown source")
            st.markdown(f"**{i}.** {source_name}")

def render_document_upload() -> object | None:
    """Render a file uploader for dynamic PDF ingestion.

    Returns:
        The uploaded file object if a file is uploaded, else None.
    """
    st.markdown("### 📄 Upload medical reports")
    st.markdown(
        "<div style='font-size: 0.85rem; color: #64748b; margin-bottom: 10px;'>"
        "Upload a medical PDF. It will be temporarily added to the AI's "
        "knowledge base for this session only."
        "</div>",
        unsafe_allow_html=True,
    )
    return st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="The file is stored in-memory for this session only.",
        label_visibility="collapsed",
    )

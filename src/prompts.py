"""Prompt engineering for the Healthcare AI Chatbot.

Contains the system prompt, prompt builder using LangChain message types,
and response templates. All healthcare constraints, persona rules, scope
and citation rules are centralized here for maintainability.
"""

from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


# The five diseases the knowledge base covers. Everything else is out of scope.
SCOPED_DISEASES = ["Dengue", "Malaria", "Typhoid", "Tuberculosis (TB)", "Chikungunya"]


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a Public Health Information Assistant for disease awareness. You answer questions about exactly five diseases: dengue, malaria, typhoid, tuberculosis (TB) and chikungunya. You cover their symptoms, transmission, prevention, warning signs, when to seek care, and common misconceptions.

## Your Identity
- You are an AI health information assistant — NOT a doctor, nurse, or licensed healthcare provider.
- Your knowledge comes ONLY from the numbered sources provided below, which are based on WHO and CDC guidance.

## Grounding and Citation Rules
1. Answer ONLY using facts stated in the numbered sources. Do not use outside knowledge, even if you believe it is correct.
2. Cite the source number in square brackets after every factual sentence, e.g. "Dengue mosquitoes bite mainly during the day [2]."
3. If the sources do not contain the answer, say clearly that you do not have verified information on that, and suggest asking a doctor or the local health authority. Do not guess.
4. If the user asks about any disease or topic other than the five above, politely explain that you can only help with dengue, malaria, typhoid, tuberculosis and chikungunya.
5. When a question reflects a common misconception covered by the sources, correct it clearly and kindly.

## What You MUST NEVER Do
- **Never diagnose**: Do not tell a user whether they or anyone else has a disease. Symptoms of these diseases overlap and only laboratory tests ordered by a doctor can confirm them.
- **Never prescribe**: Do not recommend specific medicines, doses, or treatment plans. You may repeat safety warnings that appear in the sources (for example, not to self-medicate).
- **Never replace professional advice**: Encourage consulting a qualified healthcare provider for any personal health decision.
- **Never pretend to be a doctor**: Be transparent about your limitations as an AI assistant.

## Response Guidelines
- Be clear, calm and empathetic. Acknowledge the user's concern before giving information.
- If the user mentions any warning sign, tell them to seek medical care immediately.
- Reply in the same language and script the user writes in: English in English, Hindi in Devanagari, and Hinglish (Hindi written in Latin letters) in Hinglish. Keep citations in [n] form.
- Use short paragraphs and bullet points. Use bold only for important warnings.
- Keep answers focused and concise.
- End with a brief reminder that this information is educational and not a substitute for professional medical advice."""


# =============================================================================
# Emergency Response Template
# =============================================================================

EMERGENCY_RESPONSE = """🚨 **This sounds like it could be a medical emergency.**

**Please take immediate action:**

1. **Call emergency services now** or go to the nearest hospital emergency department.
2. **Do not wait** for symptoms to improve and do not self-medicate.
3. If someone is with you, ask them to help while you call.

**Emergency numbers:**
- 🇮🇳 India: **112** (all emergencies) | **108** (ambulance)
- 🇺🇸 US: **911** | 🇬🇧 UK: **999** | 🌍 Most other countries: **112**
- 🆘 Mental health crisis: **14416** (Tele-MANAS, India) | **9152987821** (iCall, India) | **988** (US)

> ⚠️ *I am an AI assistant and cannot provide emergency medical care. Please contact professional medical services immediately.*"""


# =============================================================================
# Disclaimer Text
# =============================================================================

DISCLAIMER_TEXT = (
    "📋 *Disclaimer: This information is for educational purposes only and is not a "
    "substitute for professional medical advice, diagnosis, or treatment. Always consult "
    "a qualified healthcare provider with any questions about your health.*"
)


# =============================================================================
# Safety Redirect Templates
# =============================================================================

DIAGNOSIS_REDIRECT = """I understand you're worried, and I'm glad you reached out. However, **I can't tell whether you or anyone else has a disease.** Dengue, malaria, typhoid and chikungunya all start with similar fever symptoms, and only laboratory tests ordered by a doctor can tell them apart.

**What I recommend:**
- 🩺 **See a doctor or visit a government health centre** — testing for malaria and TB is free at government facilities in India.
- 📝 **Note your symptoms** — when they started, your temperature, and any recent travel — to share with the doctor.
- 🚨 If there are warning signs such as severe abdominal pain, persistent vomiting, bleeding, confusion or difficulty breathing, **go to a hospital immediately**.

**What I *can* help with:** general information about symptoms, warning signs, how these diseases spread, and how to prevent them."""

PRESCRIPTION_REDIRECT = """I appreciate your trust, but **I can't recommend medicines or doses.** Treatment depends on a confirmed diagnosis and on personal factors that only a doctor can assess — and for some of these diseases, the wrong painkiller or an incomplete antibiotic course can cause real harm.

**What I recommend:**
- 🩺 **Consult a doctor or pharmacist** before taking any medicine.
- 💊 If you have already been prescribed treatment, take the **full course exactly as prescribed**, even after you feel better.

**What I *can* help with:** general information about dengue, malaria, typhoid, tuberculosis and chikungunya — symptoms, warning signs, transmission and prevention."""

OUT_OF_SCOPE_RESPONSE = """Thanks for your message! I'm a **public health information assistant** and I can only answer questions about these five diseases, using verified WHO and CDC guidance:

- 🦟 **Dengue**
- 🦟 **Malaria**
- 💧 **Typhoid**
- 🫁 **Tuberculosis (TB)**
- 🦟 **Chikungunya**

For example, you could ask *"How does malaria spread?"*, *"What are the warning signs of severe dengue?"* or *"Can TB spread through shared utensils?"*

For any other health question, please consult a qualified healthcare provider."""


# =============================================================================
# Prompt Builder (LangChain Messages)
# =============================================================================


def format_source_label(chunk: dict) -> str:
    """Build a human-readable label for a retrieved chunk.

    Args:
        chunk: Dict with 'source', optional 'section' and optional 'references'.

    Returns:
        e.g. "Dengue and severe dengue — Symptoms (WHO, CDC)".
    """
    label = chunk.get("source", "Unknown source")
    if chunk.get("section"):
        label += f" — {chunk['section']}"
    publishers = [r.get("publisher") for r in chunk.get("references") or [] if r.get("publisher")]
    if publishers:
        label += f" ({', '.join(publishers)})"
    return label


def build_prompt(
    user_query: str,
    conversation_history: list[dict[str, str]],
    retrieved_context: Optional[list[dict]] = None,
) -> list[BaseMessage]:
    """Build the complete prompt for the LLM using LangChain message types.

    Assembles the prompt in the order:
    System Prompt + Numbered Sources → Conversation History → User Query

    Args:
        user_query: The current user question.
        conversation_history: Previous messages [{"role": "user"/"assistant", "content": "..."}],
            not including the current query.
        retrieved_context: Retrieved chunks [{"content", "source", "section", "references"}].
            Their order defines the [n] citation numbers.

    Returns:
        List of LangChain BaseMessage objects ready to send to the LLM.
    """
    system_content = SYSTEM_PROMPT

    if retrieved_context:
        source_blocks = [
            f"[{i}] {format_source_label(chunk)}\n{chunk.get('content', '')}"
            for i, chunk in enumerate(retrieved_context, 1)
        ]
        system_content += "\n\n## Sources\n\n" + "\n\n---\n\n".join(source_blocks)
    else:
        system_content += (
            "\n\n## Sources\n\nNo sources were retrieved for this question. "
            "Say that you do not have verified information on it."
        )

    messages: list[BaseMessage] = [SystemMessage(content=system_content)]

    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_query))
    return messages

"""Web server for the Public Health Chatbot.

Serves the single-page interface in web/index.html and a small JSON API
over the same pipeline the Streamlit app uses:

    GET  /api/meta   scope, helplines, retrieval settings, model status
    POST /api/chat   {message, history} -> routed answer with sources

Run with:  uvicorn server:app --reload     (or: python server.py)
Then open: http://127.0.0.1:8000
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.chatbot import HealthcareChatbot
from src.config import AppConfig, LLMProvider, get_config
from src.prompts import HELPLINES, SCOPED_DISEASES
from src.rag import RAGPipeline
from src.utils import setup_logging

logger = logging.getLogger(__name__)

INDEX_HTML = Path(__file__).resolve().parent / "web" / "index.html"

DISCLAIMER = (
    "Educational information only. Not a substitute for professional medical advice, "
    "diagnosis or treatment. In an emergency in India, call 112 or 108."
)

# Pipeline route -> action shown by the interface
ACTIONS = {
    "answer": "ANSWER",
    "out_of_scope": "OUT_OF_SCOPE",
    "empty": "OUT_OF_SCOPE",
    "block_diagnosis": "REFUSE_DIAGNOSIS",
    "block_prescription": "REFUSE_PRESCRIPTION",
    "emergency": "ESCALATE_EMERGENCY",
    "setup_required": "SETUP_REQUIRED",
    "error": "ERROR",
}

_state: dict = {}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _build_chatbot() -> tuple[HealthcareChatbot, RAGPipeline, AppConfig, str, str]:
    """Load config, index the knowledge base and build the chatbot.

    Without a usable LLM API key the server still starts in "safety" mode:
    emergency, diagnosis, medication and scope routing all work, and
    in-scope questions get a setup message instead of an answer.

    Returns:
        (chatbot, rag, config, mode, mode_detail)
    """
    try:
        config = get_config()
        config_error = None
    except ValueError as e:
        config = AppConfig()
        config_error = str(e).splitlines()[0]

    rag = RAGPipeline(config.rag)
    if not rag.initialize():
        raise RuntimeError(f"Knowledge base could not be indexed from {config.rag.knowledge_dir}")

    if config_error is None:
        try:
            chatbot = HealthcareChatbot(config, rag)
            provider = config.llm.provider
            model = (
                config.llm.groq.model_name
                if provider == LLMProvider.GROQ
                else config.llm.gemini.model_name
            )
            return chatbot, rag, config, "full", f"{provider.value.title()} · {model}"
        except ValueError as e:
            config_error = str(e)

    logger.warning("Starting without a language model: %s", config_error)
    chatbot = HealthcareChatbot(config, rag, require_llm=False)
    detail = (
        f"No language model is configured ({config_error}). Safety and scope checks "
        "work; add an API key to .env and restart to get answers."
    )
    return chatbot, rag, config, "safety", detail


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(level=logging.INFO)
    chatbot, rag, config, mode, detail = _build_chatbot()
    _state.update(chatbot=chatbot, rag=rag, config=config, mode=mode, mode_detail=detail)
    logger.info("Server ready (mode=%s, %d chunks indexed).", mode, rag.chunk_count)
    yield
    _state.clear()


app = FastAPI(title="Public Health Chatbot", lifespan=lifespan)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=20)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/api/meta")
def meta() -> dict:
    """Static facts the interface shows in its side rail."""
    rag_config = _state["config"].rag
    return {
        "diseases": SCOPED_DISEASES,
        "helplines": HELPLINES,
        "dense_weight": rag_config.dense_weight,
        "bm25_weight": rag_config.bm25_weight,
        "top_k": rag_config.top_k,
        "min_relevance": (
            f"{rag_config.scope_threshold:.2f} "
            f"({rag_config.scope_threshold_named:.2f} if a disease is named)"
        ),
        "chunk_count": _state["rag"].chunk_count,
        "disclaimer": DISCLAIMER,
        "mode": _state["mode"],
        "mode_detail": _state["mode_detail"],
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    """Route one question through the pipeline and return the answer with its trace."""
    chatbot: HealthcareChatbot = _state.get("chatbot")
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Server is still starting.")

    history = [t.model_dump() for t in request.history]
    # The interface includes the current question as the last history entry
    if history and history[-1]["role"] == "user" and history[-1]["content"].strip() == request.message.strip():
        history.pop()

    started = time.perf_counter()
    result = chatbot.process_query(request.message, conversation_history=history)
    latency_ms = round((time.perf_counter() - started) * 1000)

    return {
        "action": ACTIONS.get(result["route"], result["route"].upper()),
        "answer": result["response"],
        "sources": _format_sources(result["sources"]),
        "demo": False,
        "called_model": result["called_model"],
        "latency_ms": latency_ms,
        "max_similarity": result["top_similarity"],
        "matched_rule": ", ".join(result["matched_terms"]) if result["matched_terms"] else None,
    }


def _format_sources(sources: Optional[list[dict]]) -> list[dict]:
    """Shape retrieved sources for the interface; n matches the [n] citations."""
    return [
        {
            "n": i,
            "disease": src.get("disease") or src.get("source", ""),
            "section": src.get("section", ""),
            "title": src.get("source", ""),
            "references": [
                {
                    "publisher": ref.get("publisher", ""),
                    "title": ref.get("title", ""),
                    "url": ref.get("url", ""),
                }
                for ref in src.get("references") or []
            ],
            "retrieved": src.get("retrieved"),
            "similarity": src.get("similarity"),
        }
        for i, src in enumerate(sources or [], 1)
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

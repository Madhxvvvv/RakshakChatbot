# 🦟 AI-Driven Public Health Chatbot for Disease Awareness

*A Retrieval-Augmented Generation system with source citation and safety-aware refusal.*

Minor Project (ES-451), Maharaja Agrasen Institute of Technology — Madhav Kaushik, Krish Arora, Pransh Keshri. Faculty: Dr Sachin Gupta.

The chatbot answers natural-language questions about **five diseases only** — **dengue, malaria, typhoid, tuberculosis and chikungunya** — using a curated, source-attributed knowledge base built from WHO and CDC guidance. It cites the source of every factual claim, refuses requests for diagnosis or medication, refuses questions outside its knowledge base, and escalates emergencies to human medical care.

> ⚠️ **Disclaimer:** This is an awareness tool only. It does not diagnose, prescribe, or replace a qualified healthcare provider. In an emergency in India, call **112** or **108**.

---

## ✨ What it does

| In scope | Explicit non-goals |
|---|---|
| Symptoms, transmission, prevention, warning signs and care-seeking for the five diseases | Diagnosing any individual |
| Inline `[n]` citation of every factual claim, with links to WHO/CDC pages | Medicine, dose or prescription advice |
| Correcting common misconceptions (e.g. "dengue mosquitoes bite only at night") | Triage or severity scoring |
| Emergency escalation with helpline numbers | Collecting personal or health data |
| Refusing out-of-scope questions instead of answering from model memory | Any disease outside the five in scope |

---

## 🏗️ Query pipeline

```
User query
 |-> [1] SAFETY PRE-CHECK      deterministic regex rules, no model call   (src/safety.py)
 |        emergency red flag?      --> escalate + helplines, STOP
 |        diagnosis / dose request? --> refuse + referral, STOP
 |-> [2] HYBRID RETRIEVAL       0.7 × dense cosine + 0.3 × BM25           (src/rag.py)
 |-> [3] SCOPE CHECK            best similarity below floor? --> refuse, STOP
 |-> [4] GROUNDED GENERATION    top-k chunks as numbered sources;         (src/prompts.py, src/llm.py)
 |                              model bound to them, cites [n]
 |-> [5] OUTPUT VALIDATION      citation format, dosage warning, disclaimer (src/safety.py)
 v
Answer + numbered sources + disclaimer
```

**Why each decision** (see the synopsis for full justification):

- **Hybrid retrieval.** Dense embeddings catch paraphrase, and BM25 catches exact terms such as test names and helpline numbers. The corpus is small, so search is a brute-force NumPy dot product with no vector database.
- **Safety checks before any model call.** They are reproducible: the same query always routes the same way. They are fail-safe: escalation still works when the API is down. And they are auditable, because a reviewer can read the rules.
- **Scope by a similarity floor, not a topic blocklist.** If nothing in the knowledge base is close to the question, the bot refuses. Questions that name one of the five diseases use a lower floor (`RAG_SCOPE_THRESHOLD_NAMED`), because the English embedding model scores colloquial, misspelt and Hindi or Hinglish questions low. Hard negatives such as COVID-19, HIV and diabetes questions are still refused.
- **Follow-up questions.** A question like "how does *it* spread?" borrows the disease named earlier in the conversation, so it is retrieved and scoped correctly.

---

## 📁 Project structure

```
healthcare-chatbot/
├── server.py                   # FastAPI server: web/index.html + /api/meta, /api/chat
├── web/index.html              # Web chat interface (single page, no build step)
├── app.py                      # Alternative Streamlit interface
├── data/
│   ├── knowledge/              # Curated knowledge base: one Markdown file per disease
│   │   ├── dengue.md           #   YAML frontmatter: disease, title, sources (publisher/title/url),
│   │   ├── malaria.md          #   retrieved, review_status. Sections: Overview, Symptoms,
│   │   ├── typhoid.md          #   Warning signs, Transmission, Diagnosis and care, Prevention,
│   │   ├── tuberculosis.md     #   Common misconceptions, When to seek care
│   │   └── chikungunya.md
│   └── archive/                # Previous general-health PDFs (not indexed)
├── src/
│   ├── chatbot.py              # Pipeline orchestrator (safety → retrieval → scope → generation)
│   ├── config.py               # Configuration from environment variables
│   ├── document_loader.py      # Markdown (frontmatter + sections), PDF and text loaders
│   ├── embeddings.py           # Local HuggingFace sentence-transformers embeddings
│   ├── history.py              # Conversation memory (Streamlit session_state)
│   ├── llm.py                  # LangChain LLM factory (Gemini, Groq)
│   ├── prompts.py              # System prompt, numbered-source prompt builder, refusal templates
│   ├── rag.py                  # Hybrid retrieval + scope check
│   ├── safety.py               # Deterministic pre-checks and output validation
│   ├── ui.py                   # Streamlit UI components
│   └── utils.py                # Logging, input sanitisation, source formatting
└── tests/
    ├── test_prompts.py         # System prompt and prompt builder
    ├── test_rag.py             # Knowledge base, retrieval, scope check (in-scope vs hard negatives)
    ├── test_safety.py          # Emergency, diagnosis, prescription rules; output validation
    └── test_server.py          # Web API routes that need no model call
```

---

## 🚀 Setup

Requires **Python 3.12** (newer versions may lack wheels for some dependencies). With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux
uv pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add **one** API key:

```bash
GROQ_API_KEY=your_key_here      # https://console.groq.com/keys
# or
GEMINI_API_KEY=your_key_here    # https://aistudio.google.com/apikey
```

If `LLM_PROVIDER` is not set, the provider whose key is present is used. Then start the web interface:

```bash
uvicorn server:app --reload
```

and open http://127.0.0.1:8000. Each answer shows its route (answered, diagnosis or medicine request declined, out of scope, emergency), the numbered sources behind its `[n]` citations, and a trace showing whether the model was called, the latency, the best retrieval match and any safety rule that fired. Without an API key the server still starts in *safety layer only* mode.

The older Streamlit interface is still available with `streamlit run app.py`.

The first start downloads the embedding model (~90 MB). The knowledge base is re-indexed on every start, so edits to `data/knowledge/*.md` take effect after a restart.

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | auto | `gemini` or `groq`; if unset, whichever has an API key |
| `GEMINI_API_KEY` / `GEMINI_MODEL_NAME` | — / `gemini-2.5-flash` | Google Gemini |
| `GROQ_API_KEY` / `GROQ_MODEL_NAME` | — / `openai/gpt-oss-120b` | Groq |
| `TEMPERATURE` | `0.3` | Generation temperature |
| `MAX_TOKENS` | `2048` | Maximum response tokens |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model |
| `RAG_TOP_K` | `4` | Chunks given to the model as numbered sources |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1000` / `150` | Splitting of long sections |
| `RAG_DENSE_WEIGHT` / `RAG_BM25_WEIGHT` | `0.7` / `0.3` | Hybrid retrieval weights |
| `RAG_SCOPE_THRESHOLD` | `0.52` | Similarity floor for questions not naming an in-scope disease |
| `RAG_SCOPE_THRESHOLD_NAMED` | `0.20` | Similarity floor for questions naming one of the five diseases |

In the Streamlit interface, the sidebar's **Show retrieval diagnostics** toggle displays each source's similarity score.

---

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

`test_rag.py` loads the real embedding model and knowledge base. It checks that in-scope questions (including misspelt and Hinglish ones) pass the scope check, and that hard negatives such as COVID-19, HIV, diabetes, zika and cholera are refused.

---

## 💡 Example questions

- **Answered with citations:** "What are the warning signs of severe dengue?", "Do dengue mosquitoes bite only at night?", "Can TB spread through shared utensils?", "How is chikungunya different from dengue?", "TB kaise failta hai?"
- **Emergency escalation:** "My son has dengue and keeps vomiting" → emergency response with 112/108
- **Diagnosis refusal:** "Do I have malaria?", "My platelet count is 90000, is that normal?"
- **Medication refusal:** "Which medicine is best for typhoid?", "Can I take paracetamol for dengue?"
- **Out of scope:** "What are the symptoms of COVID-19?", "How to control diabetes?" → lists the five diseases it covers

---

## 🚧 Status and next steps

- **Knowledge base review:** the five disease documents were written from WHO and CDC guidance but have `review_status: pending` and `retrieved: null`. Each fact should be checked against the linked source pages, the retrieval date filled in, and the content reviewed by the faculty mentor before evaluation (Phase 2).
- **Evaluation harness (Phases 4–5):** a labelled JSONL test set across four categories (in-scope factual, emergency, diagnosis request, out-of-scope with hard negatives), metrics for routing accuracy, emergency recall, false refusal rate, groundedness and citation rate, ablations (dense-only, BM25-only, hybrid), and a threshold sweep to choose the operating point.
- **Multilingual robustness (O4):** Hindi-script questions score low with the English-only embedding model. A multilingual model such as `paraphrase-multilingual-MiniLM-L12-v2` (set `EMBEDDING_MODEL`) is worth evaluating.
- **Deviations from the synopsis stack:** embeddings are local sentence-transformers rather than the Gemini embedding API, so a Groq key alone is enough. Generation can use either Gemini or Groq.

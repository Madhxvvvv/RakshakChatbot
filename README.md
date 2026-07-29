# 🏥 Healthcare AI Assistant

An AI-powered healthcare information chatbot that provides accurate, evidence-based health information using Large Language Models. Built with **Streamlit**, **LangChain**, and powered by **Gemini 2.5 Flash** / **Groq** with a **RAG** (Retrieval-Augmented Generation) pipeline.

> ⚠️ **Disclaimer:** This chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## ✨ Features

### Core Features
- 💬 **Interactive Chat Interface** — Clean, responsive Streamlit UI with streaming responses
- 🤖 **Multi-LLM Support** — Switch between Google Gemini and Groq models in real-time
- 📚 **RAG Pipeline** — Retrieval-Augmented Generation using curated health knowledge base
- 🛡️ **Safety Guardrails** — Emergency detection, diagnosis prevention, prescription blocking
- 👤 **Health Profile** — Personalized responses based on user-provided health details
- 💾 **Conversation Memory** — Maintains context across the conversation session
- 📋 **Medical Disclaimers** — Automatic, context-appropriate health disclaimers

### Enhanced Features
- 🔄 **Streaming Responses** — Real-time token-by-token response display
- 📈 **Dynamic PDF Uploads** — Users can upload their own medical reports. The system securely parses and indexes them into an ephemeral, in-memory FAISS database for that session only.
- 📊 **Advanced PDF Parsing** — Table extraction, multi-column layout handling via PyMuPDF
- 🔍 **Source Citations** — References to knowledge base documents in responses
- 📥 **Export Chat** — Download conversation history as text file
- 🗑️ **Clear History** — One-click conversation reset
- 🎨 **Custom Styling** — Premium UI with gradients, badges, and polished layout

### Safety Features
- 🚨 **Emergency Detection** — Instantly recognizes medical emergencies and provides emergency contacts
- 🚫 **Diagnosis Prevention** — Blocks attempts to get medical diagnoses
- 💊 **Prescription Prevention** — Refuses to recommend specific medications
- 🔒 **Off-Topic Filtering** — Redirects non-health queries back to health topics
- ✅ **Output Validation** — Post-generation checks for unsafe content

---

## 🏗️ Architecture

```
User Query
    ↓
┌─────────────────────────────────┐
│       Streamlit UI (app.py)     │
│    Chat Interface + Sidebar     │
│  Health Profile + LLM Selector  │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│    Safety Guardrails (safety.py)│
│  Emergency | Diagnosis | Rx     │
│      Off-topic | Validation     │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│    RAG Pipeline (rag.py)        │
│  Document Loader → Chunking     │
│  Embeddings → FAISS Search      │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│   Prompt Builder (prompts.py)   │
│  System + Profile + RAG +       │
│  History + Query                │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│     LangChain LLM (llm.py)      │
│  Gemini 2.5 Flash | Groq        │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│   Output Validation (safety.py) │
│  Disclaimer + Safety Check      │
└────────────┬────────────────────┘
             ↓
        Response + Sources → UI
```

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit |
| **LLM Framework** | LangChain |
| **LLM Providers** | Google Gemini 2.5 Flash, Groq (Llama 3.3, Mixtral) |
| **Embeddings** | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| **Vector Store** | FAISS (via LangChain) |
| **PDF Processing** | PyMuPDF (tables, multi-column, layout-aware) |
| **Language** | Python 3.11+ |
| **Testing** | Pytest |

---

## 📁 Project Structure

```
healthcare-chatbot/
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
├── app.py                      # Main Streamlit entry point
├── README.md                   # This file
├── requirements.txt            # Python dependencies
│
├── data/
│   └── knowledge/              # RAG knowledge base documents (PDF format)
│       ├── common_symptoms.pdf
│       ├── first_aid.pdf
│       ├── healthy_lifestyle.pdf
│       ├── nutrition_guidelines.pdf
│       └── preventive_healthcare.pdf
│
├── docs/                       # Documentation
│   ├── Architecture_Presentation.pdf
│   └── Logic_Documentation.pdf
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── chatbot.py              # Core orchestrator
│   ├── config.py               # Centralized configuration
│   ├── document_loader.py      # Advanced document loading (PDF/MD/TXT)
│   ├── embeddings.py           # HuggingFace embeddings (LangChain)
│   ├── history.py              # Conversation memory (session_state)
│   ├── llm.py                  # LangChain LLM factory (Gemini + Groq)
│   ├── prompts.py              # System prompt & prompt builder
│   ├── rag.py                  # RAG pipeline orchestrator
│   ├── safety.py               # Safety guardrails (pre/post LLM)
│   ├── ui.py                   # Streamlit UI components
│   ├── utils.py                # Utility functions
│   └── vectorstore.py          # FAISS vector store (LangChain)
│
├── tests/                      # Test suite
│   ├── test_prompts.py
│   └── test_safety.py
│
└── vectorstore/                # FAISS index cache (auto-generated)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- A Google Gemini API key ([get one free](https://aistudio.google.com/apikey))
- A Groq API key (Optional) ([get one free](https://console.groq.com/keys))

### Step 1: Clone the Repository
```bash
git clone https://github.com/avi-gajera/healthcare-chatbot.git
cd healthcare-chatbot
```

### Step 2: Create Virtual Environment
```bash
python -m venv .venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy the environment template
cp .env.example .env

# Edit .env and add your API keys
# GEMINI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here  (optional)
```

### Step 5: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## ⚙️ Configuration

All configuration is managed via environment variables in the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Active LLM provider (`gemini` or `groq`) |
| `GEMINI_API_KEY` | — | Google Gemini API key (required for Gemini) |
| `GEMINI_MODEL_NAME` | `gemini-2.5-flash` | Gemini model to use |
| `GROQ_API_KEY` | — | Groq API key (required for Groq) |
| `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` | Groq model to use |
| `TEMPERATURE` | `0.3` | Generation temperature (0.0–1.0) |
| `MAX_TOKENS` | `2048` | Maximum response tokens |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RAG_TOP_K` | `5` | Number of retrieved chunks |
| `RAG_CHUNK_SIZE` | `1000` | Document chunk size |
| `RAG_CHUNK_OVERLAP` | `200` | Chunk overlap for splitting |

---

## 🧪 Testing

Run the test suite:
```bash
# Activate virtual environment first
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test Coverage
- **Safety Guardrails** — 60+ test cases covering emergency detection, diagnosis prevention, prescription blocking, off-topic filtering, edge cases, and output validation
- **Prompt Engineering** — System prompt content validation, LangChain message type checking, RAG context injection, health profile injection, full pipeline assembly

---

## 💡 Usage Examples

### General Health Questions
- "What are the symptoms of the common cold?"
- "How can I improve my diet?"
- "What exercises are good for back pain?"

### First Aid
- "How do I treat a minor burn?"
- "What should I do for a nosebleed?"
- "What is the RICE method for sprains?"

### Preventive Care
- "What vaccinations do adults need?"
- "How much water should I drink daily?"
- "How can I improve my sleep quality?"

### Safety Demonstrations
- Emergency: "I'm having chest pain and can't breathe" → Immediate emergency response
- Diagnosis: "What disease do I have?" → Polite redirect to a healthcare professional
- Prescription: "What medicine should I take?" → Educational redirect
- Off-topic: "Write me a poem" → Redirect to health topics

---

## 🛡️ Safety & Ethics

This chatbot is designed with healthcare safety as the top priority:

1. **Never Diagnoses** — Explicitly refuses to diagnose conditions
2. **Never Prescribes** — Will not recommend specific medications
3. **Emergency Detection** — Recognizes emergencies and provides emergency contacts
4. **Disclaimers** — Every health response includes an appropriate medical disclaimer
5. **Evidence-Based** — Responses grounded in WHO, CDC, NIH, and Mayo Clinic guidelines
6. **Transparency** — Always identifies itself as an AI, not a healthcare provider

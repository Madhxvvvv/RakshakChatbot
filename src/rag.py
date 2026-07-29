"""Retrieval-Augmented Generation (RAG) pipeline for the Healthcare AI Chatbot.

Uses LangChain components to orchestrate document ingestion, chunking,
embedding, and retrieval for grounding LLM responses in curated
healthcare knowledge.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import RAGConfig
from src.document_loader import load_documents, load_single_document
from src.embeddings import get_embedding_model
from src.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """A piece of retrieved context from the knowledge base.

    Attributes:
        content: The text content of the retrieved chunk.
        source: The source document name.
    """

    content: str
    source: str


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline using LangChain.

    Handles the full RAG lifecycle:
    1. Loading documents (PDF, Markdown, Text) with advanced parsing
    2. Splitting documents into overlapping chunks (LangChain text splitters)
    3. Embedding and indexing chunks in FAISS (LangChain vector store)
    4. Retrieving relevant chunks at query time

    Args:
        config: RAG configuration (embedding model, chunk size, etc.).
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embeddings = get_embedding_model(config)
        self._vectorstore = VectorStore(self._embeddings)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            is_separator_regex=False,
        )
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the RAG pipeline has been initialized with documents."""
        return self._initialized

    def initialize(self) -> bool:
        """Initialize the RAG pipeline.

        Tries to load a cached vector store first. If not available,
        ingests documents from the knowledge directory.

        Returns:
            True if initialization was successful, False otherwise.
        """
        # Try loading cached index first
        if self._vectorstore.load(self._config.vectorstore_dir):
            self._initialized = True
            logger.info(
                "RAG initialized from cache (%d vectors).",
                self._vectorstore.document_count,
            )
            return True

        # Ingest documents from knowledge directory
        return self.ingest_documents()

    def ingest_documents(self, directory: Optional[Path] = None) -> bool:
        """Ingest documents from the knowledge directory.

        Uses the advanced document loader for PDFs (tables, multi-column,
        layout-aware extraction) and LangChain text splitter for chunking.

        Args:
            directory: Override knowledge directory. Defaults to config.

        Returns:
            True if at least one document was ingested successfully.
        """
        knowledge_dir = directory or self._config.knowledge_dir

        # Step 1: Load documents using advanced loaders
        raw_documents = load_documents(knowledge_dir)
        if not raw_documents:
            logger.warning("No documents loaded from %s", knowledge_dir)
            return False

        logger.info("Loaded %d raw document(s). Splitting into chunks...", len(raw_documents))

        # Step 2: Split documents into chunks
        chunks = self._text_splitter.split_documents(raw_documents)

        if not chunks:
            logger.warning("No chunks generated after splitting.")
            return False

        logger.info("Generated %d chunks from %d document(s).", len(chunks), len(raw_documents))

        # Step 3: Embed and index chunks
        self._vectorstore.add_documents(chunks)

        # Step 4: Save to disk for caching
        self._config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore.save(self._config.vectorstore_dir)

        self._initialized = True
        logger.info(
            "RAG pipeline initialized: %d chunks indexed.",
            len(chunks),
        )
        return True

    def ingest_single_document(self, filepath: Path) -> int:
        """Ingest a single document into the active FAISS index.
        
        This method is designed for dynamic user uploads. It chunks and embeds
        the document and adds it to the active vector store, but it intentionally
        DOES NOT call .save(). This ensures the document remains ephemeral
        to the current session and is not persisted to the global knowledge base.
        
        Args:
            filepath: Path to the uploaded document.
            
        Returns:
            Number of chunks successfully indexed.
        """
        raw_documents = load_single_document(filepath)
        if not raw_documents:
            logger.warning("No documents loaded from %s", filepath)
            return 0
            
        chunks = self._text_splitter.split_documents(raw_documents)
        if not chunks:
            logger.warning("No chunks generated for %s", filepath)
            return 0
            
        # Embed and index into the IN-MEMORY vectorstore only
        self._vectorstore.add_documents(chunks)
        
        logger.info("Indexed %d ephemeral chunks from %s", len(chunks), filepath.name)
        return len(chunks)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[RetrievedContext]:
        """Retrieve relevant context for a query.

        Args:
            query: The user's question.
            top_k: Number of chunks to retrieve. Defaults to config value.

        Returns:
            List of RetrievedContext objects with content and source.
        """
        if not self._initialized:
            logger.warning("RAG pipeline not initialized. No context retrieved.")
            return []

        k = top_k or self._config.top_k
        documents = self._vectorstore.search(query, top_k=k)

        results = [
            RetrievedContext(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown source"),
            )
            for doc in documents
        ]

        logger.info(
            "Retrieved %d chunks for query: '%s...'",
            len(results),
            query[:50],
        )
        return results

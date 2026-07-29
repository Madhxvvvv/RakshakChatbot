"""FAISS vector store for the Healthcare AI Chatbot.

Uses LangChain's FAISS wrapper for similarity search over embedded
document chunks. Supports save/load for persistence across sessions.
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class VectorStore:
    """LangChain FAISS-based vector store for document similarity search.

    Wraps LangChain's FAISS implementation to provide a clean interface
    for adding documents, searching, and persisting to disk.

    Args:
        embeddings: A LangChain HuggingFaceEmbeddings instance.
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings) -> None:
        self._embeddings = embeddings
        self._store: Optional[FAISS] = None
        self._is_built = False

    @property
    def is_built(self) -> bool:
        """Whether the index has been built with documents."""
        return self._is_built

    @property
    def document_count(self) -> int:
        """Approximate number of documents in the store."""
        if self._store is None:
            return 0
        try:
            return self._store.index.ntotal
        except Exception:
            return 0

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store.

        Creates the FAISS index on first call, extends it on subsequent calls.

        Args:
            documents: List of LangChain Document objects to add.
        """
        if not documents:
            logger.warning("No documents to add to vector store.")
            return

        logger.info("Adding %d documents to vector store...", len(documents))

        if self._store is None:
            # Create new FAISS index from documents
            self._store = FAISS.from_documents(documents, self._embeddings)
        else:
            # Add to existing index
            self._store.add_documents(documents)

        self._is_built = True
        logger.info(
            "Vector store updated. Total vectors: %d",
            self.document_count,
        )

    def search(self, query: str, top_k: int = 3) -> list[Document]:
        """Search for the most similar documents to a query.

        Args:
            query: The search query text.
            top_k: Number of results to return.

        Returns:
            List of the most relevant LangChain Document objects.
        """
        if not self._is_built or self._store is None:
            logger.warning("Vector store is empty. No results returned.")
            return []

        results = self._store.similarity_search(query, k=top_k)

        for i, doc in enumerate(results):
            logger.debug(
                "Match %d: source=%s, preview=%s...",
                i + 1,
                doc.metadata.get("source", "Unknown"),
                doc.page_content[:80],
            )

        return results

    def search_with_scores(
        self, query: str, top_k: int = 3
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores.

        Args:
            query: The search query text.
            top_k: Number of results to return.

        Returns:
            List of (Document, score) tuples sorted by relevance.
        """
        if not self._is_built or self._store is None:
            return []

        return self._store.similarity_search_with_score(query, k=top_k)

    def save(self, directory: str | Path) -> None:
        """Save the vector store to disk.

        Args:
            directory: Directory path to save the index.
        """
        if self._store is None:
            logger.warning("No vector store to save.")
            return

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._store.save_local(str(directory))
        logger.info("Vector store saved to %s (%d vectors).", directory, self.document_count)

    def load(self, directory: str | Path) -> bool:
        """Load the vector store from disk.

        Args:
            directory: Directory path containing the saved index.

        Returns:
            True if loaded successfully, False otherwise.
        """
        directory = Path(directory)
        index_file = directory / "index.faiss"

        if not index_file.exists():
            logger.info("No saved vector store found at %s.", directory)
            return False

        try:
            self._store = FAISS.load_local(
                str(directory),
                self._embeddings,
                allow_dangerous_deserialization=True,
            )
            self._is_built = True
            logger.info(
                "Vector store loaded from %s (%d vectors).",
                directory,
                self.document_count,
            )
            return True
        except Exception as e:
            logger.error("Failed to load vector store: %s", e)
            return False

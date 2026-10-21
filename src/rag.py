"""Retrieval-Augmented Generation (RAG) pipeline for the Healthcare AI Chatbot.

Loads the curated disease knowledge base, splits it into section chunks,
and retrieves the most relevant chunks with hybrid search:

    score = dense_weight * cosine(query, chunk) + bm25_weight * normalised BM25

Dense retrieval catches paraphrase; BM25 catches exact terms (test names,
helpline numbers) that embeddings tend to blur. The best dense similarity is
also used as the scope check: if nothing in the knowledge base is close to the
question, the question is out of scope and the chatbot refuses. Questions that
name one of the five diseases use a lower floor, because embeddings separate
hard negatives (e.g. COVID-19 symptoms) from colloquial or code-switched
in-scope questions poorly on similarity alone.

The corpus is small (five documents), so retrieval is a brute-force NumPy dot
product over an in-memory matrix rather than a vector database.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from src.config import RAGConfig
from src.document_loader import load_documents
from src.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


@dataclass
class RetrievedContext:
    """A piece of retrieved context from the knowledge base.

    Attributes:
        content: The text content of the retrieved chunk.
        source: The source document title (e.g. "Dengue and severe dengue").
        section: The section within the document (e.g. "Symptoms").
        disease: The disease the document covers (e.g. "Dengue").
        references: Citation metadata: list of {publisher, title, url}.
        retrieved: Date the sources were retrieved, if recorded.
        similarity: Dense cosine similarity between query and chunk.
        score: Combined hybrid retrieval score.
    """

    content: str
    source: str
    section: str = ""
    disease: str = ""
    references: list[dict] = field(default_factory=list)
    retrieved: Optional[str] = None
    similarity: float = 0.0
    score: float = 0.0


@dataclass
class RetrievalResult:
    """Outcome of a retrieval call, including the scope decision.

    Attributes:
        contexts: Top-k chunks ranked by hybrid score.
        top_similarity: Highest dense cosine similarity across the corpus.
        in_scope: Whether top_similarity cleared the scope threshold.
    """

    contexts: list[RetrievedContext]
    top_similarity: float
    in_scope: bool


# Names of the five in-scope diseases, including common misspellings and
# Hindi script, so that colloquial and code-switched queries are recognised.
_SCOPED_DISEASE_RE = re.compile(
    r"\b(?:dengu\w*|mal[ae]?ria\w*|typh?(?:oi|io)d\w*|tyfoid\w*|widal|enteric\s+fever"
    r"|tuberc?u?losis|\bt\.?b\b\.?|chi[ck]+[ue]n?g[ou]n[iy]*[ae]?\w*)"
    r"|डेंगू|मलेरिया|टाइफाइड|टीबी|तपेदिक|चिकनगुनिया",
    re.IGNORECASE,
)


def mentions_scoped_disease(text: str) -> bool:
    """Whether the text names one of the five in-scope diseases."""
    return bool(_SCOPED_DISEASE_RE.search(text))


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


class RAGPipeline:
    """Hybrid (dense + BM25) retrieval over the curated knowledge base.

    Args:
        config: RAG configuration (embedding model, weights, threshold, etc.).
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embeddings = get_embedding_model(config)
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],
            is_separator_regex=False,
        )
        self._chunks: list[Document] = []
        self._matrix: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Okapi] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the pipeline has indexed the knowledge base."""
        return self._initialized

    @property
    def chunk_count(self) -> int:
        """Number of indexed chunks."""
        return len(self._chunks)

    def initialize(self, directory: Optional[Path] = None) -> bool:
        """Load, chunk and index the knowledge base.

        The index is rebuilt on every start (a few seconds for this corpus),
        so edits to the knowledge base files take effect on restart.

        Args:
            directory: Override knowledge directory. Defaults to config.

        Returns:
            True if at least one chunk was indexed.
        """
        knowledge_dir = directory or self._config.knowledge_dir

        raw_documents = load_documents(knowledge_dir)
        if not raw_documents:
            logger.warning("No documents loaded from %s", knowledge_dir)
            return False

        chunks = self._text_splitter.split_documents(raw_documents)
        # Sub-chunks of a long section lose the "<Title> — <Section>" prefix;
        # restore it so every chunk names its disease.
        for chunk in chunks:
            heading = chunk.metadata.get("heading")
            if heading and not chunk.page_content.startswith(heading):
                chunk.page_content = f"{heading}\n\n{chunk.page_content}"

        if not chunks:
            logger.warning("No chunks generated after splitting.")
            return False

        vectors = self._embeddings.embed_documents([c.page_content for c in chunks])
        self._matrix = np.asarray(vectors, dtype=np.float32)
        self._bm25 = BM25Okapi([_tokenize(c.page_content) for c in chunks])
        self._chunks = chunks
        self._initialized = True

        logger.info(
            "RAG pipeline initialized: %d chunks from %d section(s).",
            len(chunks),
            len(raw_documents),
        )
        return True

    def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Retrieve the most relevant chunks and decide whether the query is in scope.

        Args:
            query: The (possibly context-augmented) user question.
            top_k: Number of chunks to return. Defaults to config value.

        Returns:
            RetrievalResult with ranked contexts and the scope decision.
        """
        if not self._initialized or self._matrix is None or self._bm25 is None:
            logger.warning("RAG pipeline not initialized. No context retrieved.")
            return RetrievalResult(contexts=[], top_similarity=0.0, in_scope=False)

        k = top_k or self._config.top_k

        # Embeddings are L2-normalised, so the dot product is cosine similarity.
        query_vec = np.asarray(self._embeddings.embed_query(query), dtype=np.float32)
        dense = self._matrix @ query_vec

        lexical = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        if lexical.max() > 0:
            lexical = lexical / lexical.max()

        hybrid = self._config.dense_weight * dense + self._config.bm25_weight * lexical
        top_indices = np.argsort(-hybrid)[:k]

        contexts = [
            RetrievedContext(
                content=self._chunks[i].page_content,
                source=self._chunks[i].metadata.get("source", "Unknown source"),
                section=self._chunks[i].metadata.get("section", ""),
                disease=self._chunks[i].metadata.get("disease", ""),
                references=self._chunks[i].metadata.get("references", []),
                retrieved=self._chunks[i].metadata.get("retrieved"),
                similarity=float(dense[i]),
                score=float(hybrid[i]),
            )
            for i in top_indices
        ]

        top_similarity = float(dense.max())
        floor = (
            self._config.scope_threshold_named
            if mentions_scoped_disease(query)
            else self._config.scope_threshold
        )
        in_scope = top_similarity >= floor

        logger.info(
            "Retrieved %d chunks (top similarity %.3f, in_scope=%s) for: '%s...'",
            len(contexts),
            top_similarity,
            in_scope,
            query[:50],
        )
        return RetrievalResult(contexts=contexts, top_similarity=top_similarity, in_scope=in_scope)

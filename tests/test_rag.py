"""Tests for the knowledge base, hybrid retrieval and scope check.

The retrieval tests load the real embedding model (downloaded on first run)
and index the real knowledge base, so they double as a regression check on
the scope thresholds: in-scope questions must be answered and hard negatives
(other diseases) must be refused.
"""

from pathlib import Path

import pytest

from src.config import RAGConfig
from src.document_loader import load_documents
from src.rag import RAGPipeline, mentions_scoped_disease

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DISEASES = {"Dengue", "Malaria", "Typhoid", "Tuberculosis", "Chikungunya"}


# =============================================================================
# Knowledge Base
# =============================================================================


@pytest.fixture(scope="module")
def documents():
    return load_documents(KNOWLEDGE_DIR)


class TestKnowledgeBase:
    """The knowledge base must cover exactly the five diseases, with citations."""

    def test_covers_exactly_five_diseases(self, documents) -> None:
        assert {d.metadata["disease"] for d in documents} == DISEASES

    def test_every_section_cites_who_and_cdc(self, documents) -> None:
        for doc in documents:
            publishers = {r["publisher"] for r in doc.metadata["references"]}
            assert {"WHO", "CDC"} <= publishers, doc.metadata["heading"]
            assert all(r["url"].startswith("https://") for r in doc.metadata["references"])

    def test_every_disease_has_core_sections(self, documents) -> None:
        required = {"Symptoms", "Transmission", "Prevention", "Common misconceptions", "When to seek care"}
        for disease in DISEASES:
            sections = {d.metadata["section"] for d in documents if d.metadata["disease"] == disease}
            assert required <= sections, f"{disease} is missing {required - sections}"

    def test_chunks_name_their_disease(self, documents) -> None:
        for doc in documents:
            assert doc.page_content.startswith(doc.metadata["source"])


# =============================================================================
# Disease-name detection
# =============================================================================


class TestMentionsScopedDisease:
    @pytest.mark.parametrize(
        "text",
        [
            "symptoms of dengue",
            "malria symptms",
            "maleria test kaise hota hai",
            "is TB curable",
            "what is the widal test",
            "chikungunia joint pain",
            "typhiod vaccine",
            "मलेरिया कैसे फैलता है",
        ],
    )
    def test_detects_disease_names(self, text: str) -> None:
        assert mentions_scoped_disease(text)

    @pytest.mark.parametrize(
        "text",
        ["symptoms of covid", "how is HIV spread", "diabetes diet", "hello", "stable condition"],
    )
    def test_ignores_other_text(self, text: str) -> None:
        assert not mentions_scoped_disease(text)


# =============================================================================
# Retrieval + Scope Check
# =============================================================================


@pytest.fixture(scope="module")
def rag() -> RAGPipeline:
    pipeline = RAGPipeline(RAGConfig())
    assert pipeline.initialize(KNOWLEDGE_DIR)
    return pipeline


class TestRetrieval:
    @pytest.mark.parametrize(
        ("query", "expected_source", "expected_section"),
        [
            ("What are the symptoms of dengue?", "Dengue and severe dengue", "Symptoms"),
            ("How does malaria spread?", "Malaria", "Transmission"),
            ("How can I prevent typhoid?", "Typhoid fever", "Prevention"),
            ("difference between chikungunya and dengue", "Chikungunya", "Chikungunya and dengue"),
        ],
    )
    def test_top_result_is_relevant(
        self, rag: RAGPipeline, query: str, expected_source: str, expected_section: str
    ) -> None:
        top = rag.retrieve(query).contexts[0]
        assert (top.source, top.section) == (expected_source, expected_section)

    def test_bm25_finds_exact_terms(self, rag: RAGPipeline) -> None:
        """Exact terms like a helpline number should be retrievable."""
        contexts = rag.retrieve("TB helpline number").contexts
        assert any("1800-11-6666" in c.content for c in contexts)


class TestScopeCheck:
    @pytest.mark.parametrize(
        "query",
        [
            "What are the symptoms of dengue?",
            "Is TB curable?",
            "do dengue mosquitoes bite only at night",
            "can TB spread by sharing plates",
            "what is the widal test",
            "dengue ke lakshan kya hai",
            "malria symptms",
            "mosquito breeding in coolers",
        ],
    )
    def test_in_scope_questions_are_answered(self, rag: RAGPipeline, query: str) -> None:
        assert rag.retrieve(query).in_scope

    @pytest.mark.parametrize(
        "query",
        [
            # Hard negatives: health questions outside the knowledge base
            "What are the symptoms of COVID-19?",
            "How is HIV transmitted?",
            "How to control diabetes?",
            "symptoms of common cold",
            "what is zika virus",
            "what is cholera",
            "symptoms of jaundice",
            # Clearly unrelated
            "who won the cricket world cup",
            "write a python function",
            "hi",
        ],
    )
    def test_out_of_scope_questions_are_refused(self, rag: RAGPipeline, query: str) -> None:
        assert not rag.retrieve(query).in_scope

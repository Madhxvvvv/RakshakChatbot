"""Tests for the web API (server.py).

Only routes that never call the language model are exercised, so these
tests run without an API key and make no network requests to an LLM.
"""

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_index_serves_page(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "Fever questions" in res.text


def test_meta_lists_scope_and_helplines(client: TestClient) -> None:
    meta = client.get("/api/meta").json()
    assert len(meta["diseases"]) == 5
    assert any(h["number"] == "108" for h in meta["helplines"])
    assert meta["chunk_count"] > 0
    assert meta["mode"] in {"full", "safety"}


@pytest.mark.parametrize(
    ("message", "action"),
    [
        ("I am coughing up blood", "ESCALATE_EMERGENCY"),
        ("Do I have dengue?", "REFUSE_DIAGNOSIS"),
        ("Which medicine is best for typhoid?", "REFUSE_PRESCRIPTION"),
        ("How is HIV transmitted?", "OUT_OF_SCOPE"),
    ],
)
def test_routes_without_model_call(client: TestClient, message: str, action: str) -> None:
    data = client.post("/api/chat", json={"message": message, "history": []}).json()
    assert data["action"] == action
    assert data["called_model"] is False
    assert data["answer"]
    assert data["sources"] == []


def test_emergency_reports_matched_rule(client: TestClient) -> None:
    data = client.post("/api/chat", json={"message": "I am coughing up blood"}).json()
    assert data["matched_rule"]


def test_out_of_scope_reports_similarity(client: TestClient) -> None:
    data = client.post("/api/chat", json={"message": "How is HIV transmitted?"}).json()
    assert 0 < data["max_similarity"] < 1


def test_rejects_long_messages(client: TestClient) -> None:
    res = client.post("/api/chat", json={"message": "x" * 501})
    assert res.status_code == 422


def test_rejects_empty_messages(client: TestClient) -> None:
    res = client.post("/api/chat", json={"message": ""})
    assert res.status_code == 422

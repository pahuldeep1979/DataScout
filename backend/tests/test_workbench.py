import json
from unittest.mock import MagicMock

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas.rag import WebSource
from app.services import agent_graph
from app.services.demo import NOTICES
from app.services.hybrid_engine import Workspace


def pdf(*pages):
    with pymupdf.open() as doc:
        for text in pages:
            page = doc.new_page()
            if text:
                page.insert_text((50, 70), text)
        return doc.tobytes()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "workspace", Workspace(tmp_path / "workspace.db"))
    monkeypatch.setattr(main.settings, "CHROMA_PERSIST_DIR", tmp_path / "legacy")
    monkeypatch.setattr(main.settings, "GEMINI_API_KEY", "")
    monkeypatch.setattr(main.settings, "TAVILY_API_KEY", "")
    main.cache.clear()
    with TestClient(main.app) as test_client:
        yield test_client


def post_query(client, query, **kwargs):
    response = client.post("/api/query", json={"query": query, **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def upload(client, filename, data):
    return client.post("/api/documents/upload", files={"file": (filename, data, "application/pdf")})


def test_demo_conflict_checks_all_scoped_sources_even_at_top_one(client):
    question = "What is the team registration deadline?"
    result = post_query(client, question, workspace="demo", top_k=1)
    assert result["verdict"] == "conflict"
    assert len(result["chunks"]) == 1
    assert len(result["evidence"]) == 2
    for evidence in result["evidence"]:
        response = client.get("/api/demo/documents/" + evidence["source"])
        assert response.status_code == 200
        with pymupdf.open(stream=response.content) as doc:
            assert evidence["quote"] in doc[evidence["page"] - 1].get_text()

    scoped = post_query(client, question, workspace="demo", sources=[next(iter(NOTICES))], top_k=1)
    assert scoped["verdict"] == "supported"
    assert "24 November" in scoped["answer"]
    assert post_query(client, question, workspace="demo", top_k=1)["cache_hit"]


def test_rewrite_works_and_missing_questions_abstain(client):
    result = post_query(client, "How many people can be in a group?", workspace="demo")
    assert result["verdict"] == "supported"
    assert result["evidence"][0]["value"] == "3 students"
    assert sum("Rewrote" in step for step in result["trace_steps"]) == 1
    result = post_query(client, "Is there a registration fee waiver?", workspace="demo")
    assert result["verdict"] == "insufficient"
    assert result["evidence"] == []
    assert result["source_mode"] == "none"


def test_upload_replaces_stale_pages_and_scopes_are_isolated(client):
    filename = "Workshop.pdf"
    assert upload(client, filename, pdf("Event venue: Hall A", "Printing fee: INR 50")).status_code == 200
    assert post_query(client, "What is the printing fee?")["verdict"] == "supported"
    assert upload(client, filename, pdf("Event venue: Hall B")).status_code == 200
    result = post_query(client, "What is the event venue?")
    assert result["answer"] == "Event venue: Hall B"
    assert post_query(client, "What is the printing fee?")["verdict"] == "insufficient"
    assert post_query(client, "What is the event venue?", workspace="demo")["verdict"] == "insufficient"
    assert client.get("/api/documents").json() == [{"filename": filename, "chunk_count": 1, "page_count": 1}]


def test_bad_inputs_return_visible_validation_errors(client):
    assert upload(client, "Broken.pdf", b"plain text").status_code == 400
    assert upload(client, "Scanned.pdf", pdf("")).status_code == 422
    assert upload(client, "Big.pdf", b"%PDF-" + b"0" * main.MAX_UPLOAD).status_code == 413
    assert client.post("/api/query", json={"query": "  "}).status_code == 422
    assert client.post("/api/query", json={"query": "venue", "sources": ["absent.pdf"]}).status_code == 422


def test_web_requires_explicit_public_query_and_cannot_override_conflict(client, monkeypatch):
    assert upload(client, "A.pdf", pdf("Event venue: Hall A")).status_code == 200
    assert upload(client, "B.pdf", pdf("Event venue: Hall B")).status_code == 200
    monkeypatch.setattr(main.settings, "GEMINI_API_KEY", "mock")
    monkeypatch.setattr(main.settings, "TAVILY_API_KEY", "mock")
    provider = MagicMock()
    provider.__enter__.return_value.models.generate_content.return_value.text = ""
    monkeypatch.setattr(agent_graph, "gemini_client", lambda: provider)
    source = WebSource(title="Public notice", url="https://example.org/notice",
                       snippet="The expo will take place in the public library.")
    search = MagicMock(return_value=[source])
    monkeypatch.setattr(agent_graph, "search_web", search)

    assert post_query(client, "What is the event venue?", public_web_query="public venue")["verdict"] == "conflict"
    assert post_query(client, "private internal query")["verdict"] == "insufficient"
    search.assert_not_called()

    model_response = provider.__enter__.return_value.models.generate_content.return_value
    model_response.text = json.dumps({"verdict": "supported", "answer": "The expo is in the library.",
                                      "quotes": [{"source_index": 1, "quote": "invented evidence"}]})
    result = post_query(client, "private internal query", public_web_query="expo venue")
    assert result["verdict"] == "insufficient"
    search.assert_called_once_with("expo venue")

    model_response.text = json.dumps({"verdict": "supported", "answer": "The expo is in the library.",
                                      "quotes": [{"source_index": 1, "quote": source.snippet}]})
    result = post_query(client, "private internal query", public_web_query="expo venue")
    assert result["source_mode"] == "web_fallback"
    assert result["evidence"][0]["url"] == source.url


def test_legacy_chroma_chunks_are_read_once(tmp_path, monkeypatch):
    import chromadb

    legacy_path = tmp_path / "legacy"
    storage = Workspace(tmp_path / "workspace.db")
    collection = chromadb.PersistentClient(path=str(legacy_path)).create_collection(
        "verirag_documents", embedding_function=None,
    )
    collection.add(ids=["original"], embeddings=[[1.0, 0.0]], documents=["Venue: Hall A"],
                   metadatas=[{"source": "Legacy.pdf", "page": 2}])
    storage.import_legacy(legacy_path)
    storage.import_legacy(legacy_path)
    chunks, documents = storage.snapshot()
    assert len(chunks) == 1 and chunks[0].page == 2
    assert documents[0]["filename"] == "Legacy.pdf"

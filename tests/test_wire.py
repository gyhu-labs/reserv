"""Wire-format and behaviour tests with a fake backend (no OpenVINO, no GPU)."""
from __future__ import annotations

import numpy as np
from starlette.testclient import TestClient

from reserv.config import ModelSpec, Settings
from reserv.server import create_app


class FakeEmbedding:
    def encode(self, texts):
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts], dtype=np.float32), sum(len(t) for t in texts)


class FakeRerank:
    def __init__(self):
        self.calls = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        # higher score for documents sharing the longer prefix with the query
        scores = [float(len(set(query.split()) & set(d.split()))) for d in documents]
        return scores, sum(len(d) for d in documents)


def factory(spec: ModelSpec, device: str):
    return FakeEmbedding() if spec.task == "embedding" else FakeRerank()


def settings(**kw) -> Settings:
    base = {
        "device": "CPU",
        "default_model": "rerank-x",
        "max_concurrency": 4,
        "api_keys": {"hindsight": "tok-123"},
        "models": {
            "embed-x": ModelSpec(name="embed-x", task="embedding", ir="/x", tokenizer="x"),
            "rerank-x": ModelSpec(name="rerank-x", task="rerank", ir="/y", tokenizer="x"),
        },
    }
    base.update(kw)
    return Settings(**base)


AUTH = {"Authorization": "Bearer tok-123"}


def client(**kw) -> TestClient:
    return TestClient(create_app(settings(**kw), backend_factory=factory))


def test_health_and_models():
    c = client()
    h = c.get("/health").json()
    assert h["status"] == "ok" and set(h["models"]) == {"embed-x", "rerank-x"}
    m = c.get("/v1/models").json()
    assert {d["id"] for d in m["data"]} == {"embed-x", "rerank-x"}


def test_auth_required_and_enforced():
    c = client()
    assert c.post("/v1/embeddings", json={"input": "hi"}).status_code == 401
    assert c.post("/v1/embeddings", json={"input": "hi"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/v1/embeddings", json={"input": "hi"}, headers=AUTH).status_code == 200


def test_embeddings_wire_format():
    c = client()
    r = c.post("/v1/embeddings", json={"input": ["abc", "de"], "model": "embed-x"}, headers=AUTH).json()
    assert r["object"] == "list" and r["model"] == "embed-x"
    assert [d["index"] for d in r["data"]] == [0, 1]
    assert r["data"][0]["embedding"] == [3.0, 1.0, 0.0]
    assert r["usage"]["prompt_tokens"] == 5


def test_embeddings_input_validation():
    c = client()
    assert c.post("/v1/embeddings", json={"input": 5}, headers=AUTH).status_code == 400
    assert c.post("/v1/embeddings", json={"input": []}, headers=AUTH).status_code == 400
    assert c.post("/v1/embeddings", json={"input": "x", "model": "rerank-x"}, headers=AUTH).status_code == 400
    assert c.post("/v1/embeddings", json={"input": "x", "model": "nope"}, headers=AUTH).status_code == 400


def test_rerank_orders_by_score_and_honours_top_n():
    c = client()
    body = {"query": "pzu dividend", "documents": ["pzu dividend paid", "unrelated text", "pzu dividend"],
            "top_n": 2, "return_documents": True}
    r = c.post("/rerank", json=body, headers=AUTH).json()
    assert len(r["results"]) == 2
    assert r["results"][0]["relevance_score"] >= r["results"][1]["relevance_score"]
    assert r["results"][0]["document"]["text"].startswith("pzu")
    assert r["meta"]["billed_units"]["search_units"] == 1


def test_rerank_accepts_v1_path_and_texts_alias():
    c = client()
    r = c.post("/v1/rerank", json={"query": "a b", "texts": ["a b", "c"]}, headers=AUTH)
    assert r.status_code == 200 and r.json()["results"][0]["index"] == 0


def test_rerank_validation():
    c = client()
    assert c.post("/rerank", json={"documents": ["x"]}, headers=AUTH).status_code == 400
    assert c.post("/rerank", json={"query": "q", "documents": []}, headers=AUTH).status_code == 400
    assert c.post("/rerank", json={"query": "q", "documents": ["x"], "model": "embed-x"}, headers=AUTH).status_code == 400


def test_backpressure_returns_429():
    c = client(max_concurrency=0, queue_timeout=0.01)
    r = c.post("/rerank", json={"query": "q", "documents": ["x"]}, headers=AUTH)
    assert r.status_code == 429 and "retry_after" in r.json()["error"]


def test_metrics_exposed():
    c = client()
    c.post("/rerank", json={"query": "q", "documents": ["x"]}, headers=AUTH)
    body = c.get("/metrics").text
    assert "reserv_requests_total" in body and "reserv_request_seconds" in body

"""HTTP surface: OpenAI-compatible embeddings, Cohere-compatible rerank, metrics.

Design notes that are deliberate, not incidental:
- one model per process, chosen by config; `model` in a request selects among the
  loaded registry and is validated, so a typo fails loudly instead of silently
  scoring against the wrong weights;
- inference is CPU/GPU-bound and serialised by the device, so requests go through
  a semaphore with a bounded wait and answer 429+Retry-After instead of queueing
  until the client's timeout;
- scores are sigmoid-encoded for rerank by default so they look like Cohere's
  0..1; ranking is unaffected (monotone) but a score floor stays meaningful.
"""
from __future__ import annotations

import asyncio
import hmac
import time
import uuid
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from .backends import build_backend
from .config import Settings, load_settings

VERSION = "0.1.0"

REQUESTS = Counter("reserv_requests_total", "Requests", ["endpoint", "model", "key", "status"])
LATENCY = Histogram("reserv_request_seconds", "Request latency", ["endpoint", "model"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30))
QUEUE_WAIT = Histogram("reserv_queue_wait_seconds", "Time spent waiting for the device", ["endpoint"])
INFLIGHT = Gauge("reserv_inflight", "Requests holding the device")
DOCS = Counter("reserv_documents_total", "Documents embedded/reranked", ["model", "key"])
TOKENS = Counter("reserv_tokens_total", "Tokens processed", ["model", "key"])


class Service:
    def __init__(self, settings: Settings, backend_factory=build_backend) -> None:
        self.settings = settings
        self.backends: dict[str, Any] = {}
        self.sem = asyncio.Semaphore(settings.max_concurrency)
        self._factory = backend_factory

    # --- model lifecycle -------------------------------------------------
    def load(self) -> None:
        for spec in self.settings.models.values():
            self.backends[spec.name] = self._factory(spec, self.settings.device)

    def for_task(self, task: str, name: str | None) -> tuple[str, Any]:
        chosen = name or self._default_for(task)
        if chosen not in self.backends:
            raise KeyError(f"unknown model {chosen!r}; loaded: {sorted(self.backends)}")
        spec = self.settings.models[chosen]
        if spec.task != task:
            raise ValueError(f"model {chosen!r} serves {spec.task!r}, not {task!r}")
        return chosen, self.backends[chosen]

    def _default_for(self, task: str) -> str:
        for name, spec in self.settings.models.items():
            if spec.task == task and name == self.settings.default_model:
                return name
        for name, spec in self.settings.models.items():
            if spec.task == task:
                return name
        raise KeyError(f"no model serves task {task!r}")


# --- helpers -------------------------------------------------------------
def _key_name(request: Request) -> str:
    """Authenticate and return the key name, or raise KeyError."""
    settings: Settings = request.app.state.service.settings
    if not settings.api_keys:
        return "anonymous"
    header = request.headers.get("authorization") or ""
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    for name, expected in settings.api_keys.items():
        if token and hmac.compare_digest(token, expected):
            return name
    raise KeyError("unauthorized")


def _json_error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "reserv_error", **extra}}, status_code=status)


async def _guard(request: Request, endpoint: str, model: str | None):
    """Acquire the device semaphore, returning (response, None) on rejection."""
    service: Service = request.app.state.service
    started = time.perf_counter()
    try:
        await asyncio.wait_for(service.sem.acquire(), timeout=service.settings.queue_timeout)
    except TimeoutError:
        REQUESTS.labels(endpoint, model or "-", _key_name_safe(request), "429").inc()
        retry = max(1, int(service.settings.queue_timeout))
        return _json_error(429, "device busy", retry_after=retry)
    QUEUE_WAIT.labels(endpoint).observe(time.perf_counter() - started)
    INFLIGHT.inc()
    return None


def _key_name_safe(request: Request) -> str:
    try:
        return _key_name(request)
    except KeyError:
        return "-"


def _release(request: Request) -> None:
    INFLIGHT.dec()
    request.app.state.service.sem.release()


# --- endpoints -----------------------------------------------------------
async def health(request: Request) -> Response:
    service: Service = request.app.state.service
    return JSONResponse({
        "status": "ok",
        "version": VERSION,
        "device": service.settings.device,
        "models": {name: {"task": spec.task} for name, spec in service.settings.models.items()},
    })


async def models(request: Request) -> Response:
    service: Service = request.app.state.service
    return JSONResponse({"object": "list", "data": [
        {"id": name, "object": "model", "task": spec.task, "owned_by": "gyhu-labs"}
        for name, spec in service.settings.models.items()
    ]})


async def embeddings(request: Request) -> Response:
    service: Service = request.app.state.service
    try:
        key = _key_name(request)
    except KeyError:
        return _json_error(401, "invalid or missing bearer token")

    try:
        body = await request.json()
    except ValueError:
        return _json_error(400, "body must be JSON")
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        inputs = [raw_input]
    elif isinstance(raw_input, list) and all(isinstance(x, str) for x in raw_input):
        inputs = list(raw_input)
    elif isinstance(raw_input, list) and all(isinstance(x, list) for x in raw_input):
        return _json_error(400, "pre-tokenised input is not supported; send strings")
    else:
        return _json_error(400, "input must be a string or an array of strings")
    if not inputs:
        return _json_error(400, "input must not be empty")

    name = body.get("model")
    try:
        name, backend = service.for_task("embedding", name)
    except (KeyError, ValueError) as exc:
        return _json_error(400, str(exc))

    guard = await _guard(request, "embeddings", name)
    if guard is not None:
        return guard
    started = time.perf_counter()
    try:
        vectors, tokens = await asyncio.to_thread(backend.encode, inputs)
    except Exception as exc:  # noqa: BLE001 - inference failure must become a 500, not a dead worker
        REQUESTS.labels("embeddings", name, key, "500").inc()
        return _json_error(500, f"inference failed: {exc}"[:300])
    finally:
        _release(request)

    REQUESTS.labels("embeddings", name, key, "200").inc()
    LATENCY.labels("embeddings", name).observe(time.perf_counter() - started)
    DOCS.labels(name, key).inc(len(inputs))
    TOKENS.labels(name, key).inc(tokens)
    return JSONResponse({
        "object": "list",
        "model": name,
        "data": [{"object": "embedding", "index": i, "embedding": vec.astype(float).tolist()}
                 for i, vec in enumerate(vectors)],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    })


async def rerank(request: Request) -> Response:
    service: Service = request.app.state.service
    try:
        key = _key_name(request)
    except KeyError:
        return _json_error(401, "invalid or missing bearer token")

    try:
        body = await request.json()
    except ValueError:
        return _json_error(400, "body must be JSON")
    query = body.get("query")
    documents = body.get("documents") or body.get("texts") or []
    if not isinstance(query, str) or not query:
        return _json_error(400, "query must be a non-empty string")
    if isinstance(documents, list) and documents and isinstance(documents[0], dict):
        documents = [d.get("text", "") for d in documents]
    if not isinstance(documents, list) or not all(isinstance(d, str) for d in documents):
        return _json_error(400, "documents must be an array of strings")
    if not documents:
        return _json_error(400, "documents must not be empty")

    name = body.get("model")
    try:
        name, backend = service.for_task("rerank", name)
    except (KeyError, ValueError) as exc:
        return _json_error(400, str(exc))

    guard = await _guard(request, "rerank", name)
    if guard is not None:
        return guard
    started = time.perf_counter()
    try:
        scores, tokens = await asyncio.to_thread(backend.score, query, documents)
    except Exception as exc:  # noqa: BLE001 - see embeddings
        REQUESTS.labels("rerank", name, key, "500").inc()
        return _json_error(500, f"inference failed: {exc}"[:300])
    finally:
        _release(request)

    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_n = body.get("top_n") or len(order)
    want_docs = bool(body.get("return_documents"))
    results = []
    for rank, idx in enumerate(order[: int(top_n)]):
        item: dict[str, Any] = {"index": idx, "relevance_score": float(scores[idx]), "rank": rank}
        if want_docs:
            item["document"] = {"text": documents[idx]}
        results.append(item)

    REQUESTS.labels("rerank", name, key, "200").inc()
    LATENCY.labels("rerank", name).observe(time.perf_counter() - started)
    DOCS.labels(name, key).inc(len(documents))
    TOKENS.labels(name, key).inc(tokens)
    return JSONResponse({
        "id": f"rerank-{uuid.uuid4()}",
        "model": name,
        "results": results,
        "meta": {"billed_units": {"search_units": max(1, len(documents) // 100)},
                 "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                 "tokens": tokens},
    })


async def metrics(request: Request) -> Response:
    service: Service = request.app.state.service
    if not service.settings.metrics_enabled:
        return PlainTextResponse("metrics disabled\n", status_code=404)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def create_app(settings: Settings | None = None, backend_factory=build_backend, preload: bool = True) -> Starlette:
    settings = settings or load_settings()
    service = Service(settings, backend_factory)
    if preload:
        service.load()
    routes = [
        Route("/health", health),
        Route("/healthz", health),
        Route("/v1/models", models),
        Route("/v1/embeddings", embeddings, methods=["POST"]),
        Route("/rerank", rerank, methods=["POST"]),
        Route("/v1/rerank", rerank, methods=["POST"]),
    ]
    if settings.metrics_enabled:
        routes.append(Route("/metrics", metrics))
    app = Starlette(routes=routes)
    app.state.service = service
    return app


def app() -> Starlette:  # uvicorn reserv.server:app
    return create_app()

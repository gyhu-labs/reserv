# reserv — OpenVINO encoder inference service

Embeddings and cross-encoder reranking behind **standard wire formats**, so any
client works without a bespoke SDK:

| endpoint | format | typical clients |
|---|---|---|
| `POST /v1/embeddings` | OpenAI | OpenAI SDK, LangChain, Hindsight (`EMBEDDINGS_PROVIDER=openai`) |
| `POST /rerank`, `POST /v1/rerank` | Cohere | Cohere clients, Hindsight (`RERANKER_PROVIDER=cohere` + `COHERE_BASE_URL`) |
| `GET /v1/models`, `/health`, `/metrics` | OpenAI-ish / Prometheus | routing, probes, monitoring |

Nothing about a particular model is hardcoded: models are declared in
`models.yaml` (task, IR path, tokenizer, pooling, activation, batch size), so the
same image serves any converted OpenVINO encoder.

## Why it exists

Measured on an Intel Arc A310 (4 GB, single command stream), 50 rerank candidates:

| backend | per recall |
|---|---|
| OpenVINO on the Arc | **0.75–0.94 s** |
| CPU, 4 threads (i5-9400) | 7.3 s |
| CPU, OpenVINO | 11.3 s |

Embeddings run at **122 docs/s** on the same card (vs 3.4 docs/s on CPU), so a
250k-document re-embed takes ~36 min instead of ~20 h. fp16 weights halve the
footprint; they do **not** speed anything up — the GPU plugin already executes in
fp16, and forcing f32 costs 3.4×.

## Run

```bash
docker run --rm -p 8080:8080 \
  --device /dev/dri -e RESERV_DEVICE=GPU \
  -e RESERV_API_KEYS="client=secret" \
  ghcr.io/gyhu-labs/reserv:latest-bge
```

```bash
curl -s localhost:8080/rerank -H 'Authorization: Bearer secret' \
  -H 'Content-Type: application/json' \
  -d '{"query":"dywidenda PZU","documents":["PZU dividend paid 08.10","attic floor 71 m2"],"top_n":2}'
```

## Configuration

`models.yaml` (or `RESERV_CONFIG`; JSON via `RESERV_CONFIG_JSON`):

```yaml
device: GPU                 # GPU | CPU | AUTO
default_model: bge-reranker-v2-m3
max_concurrency: 1          # the device serialises; more only queues
queue_timeout: 30.0         # seconds of wait before 429 + Retry-After
models:
  bge-m3:
    task: embedding
    ir: /models/bge-m3/openvino_model.xml
    tokenizer: BAAI/bge-m3
    pooling: mean           # mean | cls
    normalize: true
  bge-reranker-v2-m3:
    task: rerank
    ir: /models/bge-reranker-v2-m3/openvino_model.xml
    tokenizer: BAAI/bge-reranker-v2-m3
    activate: sigmoid       # sigmoid | none
```

Environment: `RESERV_DEVICE`, `RESERV_API_KEYS` (`name=token,...`; empty disables
auth), `RESERV_CONFIG`, `RESERV_PORT`, `RESERV_MAX_CONCURRENCY`, `RESERV_QUEUE_TIMEOUT`.

## Operations

- `/metrics`: `reserv_requests_total{endpoint,model,key,status}`,
  `reserv_request_seconds` (histogram), `reserv_queue_wait_seconds`,
  `reserv_inflight`, `reserv_documents_total`, `reserv_tokens_total`.
- Backpressure is explicit: when the device is saturated past `queue_timeout`
  the service answers **429** with `retry_after` instead of piling up requests.
- One model per process is the intended deployment; a registry with several
  models is supported for tests and small setups.

## Build

`models.yaml` points at IRs under `/models`; CI converts them from the upstream
weights and gates the conversion on parity against reference scores
(`scripts/parity.py`, Spearman ≥ 0.999 and top-10 identical).

```bash
make models      # convert with optimum-intel (needs torch on the build host)
make image       # docker buildx build
make test        # pytest + ruff
```

## Licence

MIT for this repository. Baked models keep their own licences — see `NOTICE`.

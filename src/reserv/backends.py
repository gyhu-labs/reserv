"""OpenVINO inference backends: embeddings (mean/cls pooling) and cross-encoder rerank."""
from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

from .config import ModelSpec


class _Base:
    def __init__(self, spec: ModelSpec, device: str) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self.spec = spec
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer)
        core = ov.Core()
        self.compiled = core.compile_model(spec.ir, device)
        self.inputs = {i.get_any_name() for i in self.compiled.inputs}
        self.output = self.compiled.output(0)
        # OpenVINO may expose several outputs; pick the tensor by name when possible.
        names = [o.get_any_name() for o in self.compiled.outputs]

        def pick(preferred: Sequence[str], fallback):
            for p in preferred:
                if p in names:
                    return self.compiled.output(p)
            return fallback

        self.output = pick(("last_hidden_state", "logits", "sentence_embedding"), self.output)

    def _feed(self, enc: dict) -> dict:
        feed = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = enc.get("token_type_ids", np.zeros_like(enc["input_ids"]))
        return feed

    def token_count(self, enc: dict) -> int:
        return int(np.asarray(enc["attention_mask"]).sum())


class EmbeddingBackend(_Base):
    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, int]:
        vectors: list[np.ndarray] = []
        tokens = 0
        for i in range(0, len(texts), self.spec.batch_size):
            chunk = list(texts[i : i + self.spec.batch_size])
            enc = self.tokenizer(
                chunk, padding=True, truncation=True, max_length=self.spec.max_length, return_tensors="np"
            )
            tokens += self.token_count(enc)
            with self.lock:
                out = np.asarray(self.compiled(self._feed(enc))[self.output])
            if out.ndim == 3:  # (batch, seq, hidden) -> pool
                mask = np.asarray(enc["attention_mask"])[..., None]
                if self.spec.pooling == "cls":
                    vec = out[:, 0, :]
                else:
                    vec = (out * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
            else:
                vec = out
            vectors.append(np.asarray(vec, dtype=np.float32))
        arr = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype=np.float32)
        if self.spec.normalize and arr.size:
            arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
        return arr, tokens


class RerankBackend(_Base):
    def score(self, query: str, documents: Sequence[str]) -> tuple[list[float], int]:
        scores: list[float] = []
        tokens = 0
        for i in range(0, len(documents), self.spec.batch_size):
            chunk = list(documents[i : i + self.spec.batch_size])
            enc = self.tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="np",
            )
            tokens += self.token_count(enc)
            with self.lock:
                out = np.asarray(self.compiled(self._feed(enc))[self.output])
            flat = out.reshape(out.shape[0], -1)[:, 0].astype(np.float64).tolist()
            scores.extend(flat)
        if self.spec.activate == "sigmoid":
            scores = [float(1.0 / (1.0 + np.exp(-s))) for s in scores]
        return scores, tokens


def build_backend(spec: ModelSpec, device: str) -> _Base:
    if spec.task == "embedding":
        return EmbeddingBackend(spec, device)
    return RerankBackend(spec, device)

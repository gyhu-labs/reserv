"""Model registry and runtime configuration.

A model entry describes *how to serve* a converted OpenVINO IR; nothing about a
specific model family is hardcoded. Selection is by name from the request.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS = ("embedding", "rerank")


@dataclass
class ModelSpec:
    name: str
    task: str
    ir: str
    tokenizer: str
    max_length: int = 512
    batch_size: int = 16
    pooling: str = "mean"          # embedding only: mean | cls
    normalize: bool = True         # embedding only: L2-normalise output
    activate: str = "sigmoid"      # rerank only: sigmoid | none
    top_k_default: int | None = None

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"model {self.name!r}: task must be one of {TASKS}, got {self.task!r}")
        if self.task == "embedding" and self.pooling not in ("mean", "cls"):
            raise ValueError(f"model {self.name!r}: pooling must be mean|cls")


@dataclass
class Settings:
    device: str = "GPU"
    host: str = "0.0.0.0"
    port: int = 8080
    default_model: str | None = None
    max_concurrency: int = 1
    queue_timeout: float = 30.0
    metrics_enabled: bool = True
    api_keys: dict[str, str] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)


def parse_api_keys(raw: str | None) -> dict[str, str]:
    """`name=token,name2=token2` -> {name: token}. Empty string means auth disabled."""
    keys: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError("RESERV_API_KEYS entries must look like name=token")
        name, token = part.split("=", 1)
        keys[name.strip()] = token.strip()
    return keys


def load_settings(
    config_path: str | os.PathLike[str] | None = None, env: dict[str, str] | None = None
) -> Settings:
    env = dict(os.environ if env is None else env)
    cfg_path = Path(config_path or env.get("RESERV_CONFIG", "/etc/reserv/models.yaml"))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text()) or {}
    elif env.get("RESERV_CONFIG_JSON"):
        raw = json.loads(env["RESERV_CONFIG_JSON"])

    models: dict[str, ModelSpec] = {}
    for name, spec in (raw.get("models") or {}).items():
        models[name] = ModelSpec(name=name, **spec)

    s = Settings(
        device=env.get("RESERV_DEVICE", raw.get("device", "GPU")),
        host=env.get("RESERV_HOST", raw.get("host", "0.0.0.0")),
        port=int(env.get("RESERV_PORT", raw.get("port", 8080))),
        default_model=env.get("RESERV_DEFAULT_MODEL", raw.get("default_model")),
        max_concurrency=int(env.get("RESERV_MAX_CONCURRENCY", raw.get("max_concurrency", 1))),
        queue_timeout=float(env.get("RESERV_QUEUE_TIMEOUT", raw.get("queue_timeout", 30.0))),
        metrics_enabled=str(env.get("RESERV_METRICS", raw.get("metrics", True))).lower() not in ("0", "false", "no"),
        api_keys=parse_api_keys(env.get("RESERV_API_KEYS")),
        models=models,
    )
    if not s.models:
        raise ValueError(f"no models configured (looked at {cfg_path})")
    if s.default_model is None:
        s.default_model = next(iter(s.models))
    if s.default_model not in s.models:
        raise ValueError(f"default_model {s.default_model!r} is not in models")
    return s

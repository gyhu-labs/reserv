"""Configuration contracts that once failed silently in production use."""
import json
from pathlib import Path

import yaml

from reserv.config import load_settings

ROOT = Path(__file__).resolve().parents[1]


def test_inline_config_wins_over_baked_file(tmp_path):
    # The image always bakes /etc/reserv/models.yaml; an explicit RESERV_CONFIG_JSON must win,
    # or an embedder-only CPU deployment silently starts the baked GPU config.
    baked = tmp_path / "models.yaml"
    baked.write_text(yaml.safe_dump({"device": "GPU", "models": {
        "rr": {"task": "rerank", "ir": "/m/rr.xml", "tokenizer": "/m/rr"}}}))
    inline = {"device": "CPU", "models": {
        "emb": {"task": "embedding", "ir": "/m/emb.xml", "tokenizer": "/m/emb", "pooling": "cls"}}}
    s = load_settings(env={"RESERV_CONFIG": str(baked), "RESERV_CONFIG_JSON": json.dumps(inline)})
    assert s.device == "CPU"
    assert set(s.models) == {"emb"}


def test_baked_file_used_without_inline_config(tmp_path):
    baked = tmp_path / "models.yaml"
    baked.write_text(yaml.safe_dump({"device": "GPU", "models": {
        "rr": {"task": "rerank", "ir": "/m/rr.xml", "tokenizer": "/m/rr"}}}))
    s = load_settings(env={"RESERV_CONFIG": str(baked)})
    assert s.device == "GPU" and set(s.models) == {"rr"}


def test_served_bge_m3_pooling_matches_reference():
    # BAAI/bge-m3 is CLS-pooled (1_Pooling/config.json). Serving it mean-pooled gave vectors at
    # cosine 0.755 to the real model while the parity gate passed, because the goldens were
    # mean-pooled too. The goldens now record the model's own pooling; the served config must match.
    served = yaml.safe_load((ROOT / "models.yaml").read_text())["models"]["bge-m3"]["pooling"]
    golden = json.loads((ROOT / "fixtures" / "golden_embed.json").read_text())
    assert golden.get("pooling") == "cls"
    assert served == golden["pooling"]

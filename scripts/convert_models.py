"""Convert the upstream weights to OpenVINO IR for baking into the image.

Uses `openvino.convert_model` directly instead of optimum-intel: the conversion is
the same traced graph for encoder models, it needs no extra dependency, and the
result is gated by `scripts/parity.py` against torch fp32 references, so a
converter regression cannot slip through silently.

Requires torch + transformers on the build host (CI installs the CPU wheels).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = {
    "bge-m3": {"kind": "embedding", "repo": "BAAI/bge-m3"},
    "bge-reranker-v2-m3": {"kind": "rerank", "repo": "BAAI/bge-reranker-v2-m3"},
}
EXAMPLE = {"input_ids": [[1] * 16], "attention_mask": [[1] * 16]}


def _load(kind: str, repo: str):
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification

    if kind == "embedding":
        model = AutoModel.from_pretrained(repo, torch_dtype=torch.float32)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(repo, num_labels=1, torch_dtype=torch.float32)
    return model.eval()


def convert(name: str, out_dir: Path, weight_format: str) -> None:
    import openvino as ov
    import torch

    spec = MODELS[name]
    model = _load(spec["kind"], spec["repo"])
    example = {k: torch.tensor(v, dtype=torch.long) for k, v in EXAMPLE.items()}
    with torch.no_grad():
        ov_model = ov.convert_model(model, example_input=example)
    # batch and sequence length are dynamic; the server pads per request
    ov_model.reshape({"input_ids": [-1, -1], "attention_mask": [-1, -1]})
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "openvino_model.xml"
    ov.save_model(ov_model, target, compress_to_fp16=(weight_format == "fp16"))
    size_mb = (out_dir / "openvino_model.bin").stat().st_size / 1e6
    print(f"converted {name} -> {target} ({size_mb:.0f} MB, {weight_format})", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/models", help="output root")
    ap.add_argument("--weight-format", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    root = Path(args.out)
    for name in (MODELS if args.only is None else [args.only]):
        convert(name, root / name, args.weight_format)
    print("done ->", root.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())

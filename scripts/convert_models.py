"""Convert the upstream weights to OpenVINO IR (fp16) for baking into the image.

Run on a machine with torch available (`pip install optimum-intel[openvino] torch`).
CI runs the same code, then `scripts/parity.py` gates the result.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MODELS = {
    "bge-m3": {"kind": "embedding", "repo": "BAAI/bge-m3"},
    "bge-reranker-v2-m3": {"kind": "rerank", "repo": "BAAI/bge-reranker-v2-m3"},
}


def convert(name: str, out_dir: Path, weight_format: str) -> None:
    repo = MODELS[name]["repo"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", repo,
        "--weight-format", weight_format,
        "--task", "feature-extraction" if MODELS[name]["kind"] == "embedding" else "text-classification",
        str(out_dir),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/models", help="output root")
    ap.add_argument("--weight-format", default="fp16", choices=["fp16", "fp32", "int8"])
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    root = Path(args.out)
    for name in (MODELS if args.only is None else [args.only]):
        convert(name, root / name, args.weight_format)
    print("done ->", root.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())

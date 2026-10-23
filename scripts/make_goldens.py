"""Generate the reference (torch fp32) outputs the conversion gate compares against.

Run once on a machine with torch; commit the two JSON files it writes. CI does not
need torch: it converts to IR and compares against these references, which is what
makes the gate cheap and still meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

RERANK = "BAAI/bge-reranker-v2-m3"
EMBED = "BAAI/bge-m3"


def model_pooling(repo: str) -> str:
    """The pooling the model publishes in its sentence-transformers config: "cls" or "mean".

    The reference must not hard-code a pooling: the served config is checked against it,
    so a guess made here would be confirmed by the gate instead of caught (reserv once
    served bge-m3 mean-pooled while the model is CLS-pooled; cosine 0.755 to the real vectors).
    """
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(repo, "1_Pooling/config.json")).read_text())
    if cfg.get("pooling_mode_cls_token"):
        return "cls"
    if cfg.get("pooling_mode_mean_tokens"):
        return "mean"
    raise SystemExit(f"{repo}: unsupported pooling config {cfg}")


def rerank_logits(pairs: list[list[str]]) -> list[float]:
    tok = AutoTokenizer.from_pretrained(RERANK)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK, num_labels=1, torch_dtype=torch.float32).eval()
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), 8):
            chunk = pairs[i : i + 8]
            enc = tok([p[0] for p in chunk], [p[1] for p in chunk], padding=True, truncation=True,
                      max_length=512, return_tensors="pt")
            out.extend(model(**enc).logits.view(-1).tolist())
    return out


def embed_vectors(texts: list[str], pooling: str) -> list[list[float]]:
    tok = AutoTokenizer.from_pretrained(EMBED)
    model = AutoModel.from_pretrained(EMBED, torch_dtype=torch.float32).eval()
    vecs: list[list[float]] = []
    with torch.no_grad():
        for i in range(0, len(texts), 8):
            enc = tok(texts[i : i + 8], padding=True, truncation=True, max_length=512, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = enc["attention_mask"][..., None].float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vecs.extend(pooled.tolist())
    return vecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default="fixtures")
    ap.add_argument("--out", default="fixtures")
    ap.add_argument("--embed-only", action="store_true", help="regenerate golden_embed.json only")
    args = ap.parse_args()
    fx, out = Path(args.fixtures), Path(args.out)

    pairs = json.loads((fx / "rerank_pairs.json").read_text())["pairs"]
    texts = json.loads((fx / "embed_texts.json").read_text())["texts"]

    if not args.embed_only:
        rerank = rerank_logits(pairs)
        (out / "golden_rerank.json").write_text(json.dumps({
            "model": RERANK, "dtype": "fp32", "n": len(rerank),
            "logits": [round(float(x), 6) for x in rerank]}, indent=1))
        print(f"rerank refs: {len(rerank)}")

    pooling = model_pooling(EMBED)
    vecs = embed_vectors(texts, pooling)
    (out / "golden_embed.json").write_text(json.dumps({
        "model": EMBED, "dtype": "fp32", "dim": len(vecs[0]), "pooling": pooling,
        "vectors": [[round(float(v), 6) for v in row] for row in vecs]}, indent=1))
    print(f"embed refs: {len(vecs)}x{len(vecs[0])} pooling={pooling}")
    _ = np  # keep numpy import meaningful for future checks
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

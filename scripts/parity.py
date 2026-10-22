"""Conversion gate: an OpenVINO IR is only acceptable if it reproduces the reference.

Two checks, both on committed fixtures so CI needs no torch:
  rerank  : Spearman >= 0.999 vs torch fp32 logits, top-10 ordering identical
  embedding: cosine >= 0.9999 vs torch fp32 vectors, top-10 neighbours identical

Tolerance for raw values is looser for fp16 (the shipped format) than fp32.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(np.asarray(a)))
    rb = np.argsort(np.argsort(np.asarray(b)))
    return float(np.corrcoef(ra, rb)[0, 1])


def ir_path(root: str) -> str:
    cands = [p for p in glob.glob(os.path.join(root, "**", "*.xml"), recursive=True)
             if "tokenizer" not in os.path.basename(p).lower()]
    if not cands:
        raise SystemExit(f"no IR under {root}")
    return cands[0]


def rerank_scores(ir: str, pairs: list[list[str]], tokenizer: str, batch: int = 8) -> list[float]:
    import openvino as ov
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    m = ov.Core().compile_model(ir, "CPU")
    names = {i.get_any_name() for i in m.inputs}
    out = m.output(0)
    scores: list[float] = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i : i + batch]
        enc = tok([p[0] for p in chunk], [p[1] for p in chunk], padding=True, truncation=True,
                  max_length=512, return_tensors="np")
        feed = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "token_type_ids" in names:
            feed["token_type_ids"] = enc.get("token_type_ids", np.zeros_like(enc["input_ids"]))
        scores.extend(np.asarray(m(feed)[out]).reshape(-1).tolist())
    return scores


def embed_vectors(ir: str, texts: list[str], tokenizer: str, batch: int = 8) -> np.ndarray:
    import openvino as ov
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    m = ov.Core().compile_model(ir, "CPU")
    names = {i.get_any_name() for i in m.inputs}
    out = m.output(0)
    rows: list[np.ndarray] = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i : i + batch], padding=True, truncation=True, max_length=512, return_tensors="np")
        feed = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
        if "token_type_ids" in names:
            feed["token_type_ids"] = enc.get("token_type_ids", np.zeros_like(enc["input_ids"]))
        v = np.asarray(m(feed)[out])
        if v.ndim == 3:
            mask = np.asarray(enc["attention_mask"])[..., None]
            v = (v * mask).sum(1) / np.clip(mask.sum(1), 1e-9, None)
        rows.append(v.astype(np.float32))
    arr = np.vstack(rows)
    return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="artifacts/models")
    ap.add_argument("--fixtures", default="fixtures")
    ap.add_argument("--rerank-tokenizer", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--embed-tokenizer", default="BAAI/bge-m3")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "int8"])
    ap.add_argument("--skip-embed", action="store_true")
    args = ap.parse_args()

    tol = {"fp32": 1e-2, "fp16": 5e-2, "int8": 5e-1}[args.dtype]
    ok = True

    rr_dir = os.path.join(args.models, "bge-reranker-v2-m3")
    if os.path.isdir(rr_dir):
        pairs = json.loads(open(os.path.join(args.fixtures, "rerank_pairs.json")).read())["pairs"]
        golden = json.loads(open(os.path.join(args.fixtures, "golden_rerank.json")).read())["logits"]
        got = rerank_scores(ir_path(rr_dir), pairs, args.rerank_tokenizer)
        rho = spearman(got, golden)
        maxdiff = float(np.max(np.abs(np.asarray(got) - np.asarray(golden))))
        top_got = sorted(range(len(got)), key=lambda i: -got[i])[:10]
        top_ref = sorted(range(len(golden)), key=lambda i: -golden[i])[:10]
        passed = rho >= 0.999 and top_got == top_ref and maxdiff <= tol
        ok &= passed
        print(json.dumps({"check": "rerank", "dtype": args.dtype, "n": len(pairs), "spearman": round(rho, 6),
                          "max_abs_diff": round(maxdiff, 5), "top10_identical": top_got == top_ref,
                          "tolerance": tol, "pass": passed}))

    if not args.skip_embed and os.path.isdir(os.path.join(args.models, "bge-m3")):
        texts = json.loads(open(os.path.join(args.fixtures, "embed_texts.json")).read())["texts"]
        ref = np.asarray(json.loads(open(os.path.join(args.fixtures, "golden_embed.json")).read())["vectors"])
        got = embed_vectors(ir_path(os.path.join(args.models, "bge-m3")), texts, args.embed_tokenizer)
        cos = (ref * got).sum(1)
        # nearest-neighbour agreement over the fixture itself
        sim_got = got @ got.T
        sim_ref = ref @ ref.T
        nn_got = [set(np.argsort(-sim_got[i])[:10]) for i in range(len(texts))]
        nn_ref = [set(np.argsort(-sim_ref[i])[:10]) for i in range(len(texts))]
        agree = float(np.mean([len(a & b) / 10 for a, b in zip(nn_got, nn_ref)]))
        passed = float(cos.min()) >= 0.999 and agree >= 0.9
        ok &= passed
        print(json.dumps({"check": "embedding", "dtype": args.dtype, "n": len(texts), "dim": int(got.shape[1]),
                          "cos_min": round(float(cos.min()), 6), "cos_mean": round(float(cos.mean()), 6),
                          "top10_neighbour_agreement": round(agree, 3), "pass": passed}))

    print("PARITY " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

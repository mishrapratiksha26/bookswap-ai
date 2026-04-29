"""
Re-ranking ablation — thesis Chapter 5 Table 3.

Research question (RQ1a): Does the proposed weight configuration α=0.35,
β=0.25, γ=0.15, δ=0.15, ε=0.10 produce better top-k rankings than
alternative configurations?

Method
------
For each of 5 weight configurations × each of 21 retrieval queries:
  1. Pull the top-20 candidates from semantic search (fixed across configs —
     only the re-ranker changes).
  2. Apply rerank_books(..., weights=config).
  3. Record top-5 quality signals:
       - mean semantic similarity   (did we keep relevant books?)
       - availability@5             (what fraction are borrowable?)
       - mean rating_norm           (did we promote well-rated titles?)
       - mean popularity            (are we over-indexing on popular?)
       - top-1 available            (binary — is the very top recommendation
                                      actually borrowable?)
  4. Jaccard overlap with Proposed (C4) — quantifies how much each config
     diverges from the thesis's chosen configuration.

Why this is offline (no LLM, no network)
----------------------------------------
The ablation measures the re-ranker's direct output. Running the full
agent 5 × 35 times would cost 175 LLM calls and inject LLM noise into
what is a pure-retrieval question. Offline evaluation is cheaper, faster,
and reproducible — temperature=0 equivalent for a non-LLM component.

Usage
-----
    python scripts/eval_reranking_ablation.py
    python scripts/eval_reranking_ablation.py --out experiments/reranking_ablation.csv

Requires MongoDB reachable via DB_URL env var. No Groq calls.
"""
from __future__ import annotations

# --- Path bootstrap so `python scripts/foo.py` resolves `from app...` ----
import os, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# -------------------------------------------------------------------------

import argparse
import csv
import json
from statistics import mean
from datetime import datetime, timezone

from pymongo import MongoClient
from dotenv import load_dotenv

from app.search import search_books, rerank_books
from app.eval_queries import RETRIEVAL_QUERIES

load_dotenv()

# -------------------------------------------------------------------------
# The 5 weight configurations (thesis Chapter 5, Table 3 rows).
# Each row is (id, label, weight dict). Sums to 1.0 for every row — the
# constraint makes the configs directly comparable as convex combinations.
# -------------------------------------------------------------------------
CONFIGS: list[tuple[str, str, dict]] = [
    ("C1", "Uniform",
     {"alpha": 0.20, "beta": 0.20, "gamma": 0.20, "delta": 0.20, "epsilon": 0.20}),

    ("C2", "Semantic-only",
     {"alpha": 0.70, "beta": 0.10, "gamma": 0.10, "delta": 0.05, "epsilon": 0.05}),

    ("C3", "Availability-focused",
     {"alpha": 0.40, "beta": 0.35, "gamma": 0.10, "delta": 0.10, "epsilon": 0.05}),

    ("C4", "Proposed",
     {"alpha": 0.35, "beta": 0.25, "gamma": 0.15, "delta": 0.15, "epsilon": 0.10}),

    ("C5", "Personalisation-heavy",
     {"alpha": 0.20, "beta": 0.15, "gamma": 0.10, "delta": 0.45, "epsilon": 0.10}),
]

TOP_K = 5           # number of books the agent actually shows a user
CAND_POOL = 20      # semantic search candidate pool size before re-rank


# -------------------------------------------------------------------------
# Core measurement: per (config, query) return a row of quality signals.
# -------------------------------------------------------------------------
def measure(config_weights: dict, candidates: list[dict]) -> dict:
    """Apply rerank with these weights to the pre-fetched candidate pool
    and return top-k aggregate quality numbers. `candidates` is mutated
    in place by rerank_books — caller passes a fresh copy per config."""
    reranked = rerank_books(list(candidates), taste_vector=None, weights=config_weights)
    top = reranked[:TOP_K]
    if not top:
        return {k: 0.0 for k in ("mean_sim", "avail_at_k", "mean_rating",
                                 "mean_pop", "top1_available")}

    comps = [b.get("_score_components", {}) for b in top]
    return {
        "mean_sim":        round(mean(c.get("sim", 0.0) for c in comps), 4),
        "avail_at_k":      round(mean(c.get("avail", 0.0) for c in comps), 4),
        "mean_rating":     round(mean(c.get("rating", 0.0) for c in comps), 4),
        "mean_pop":        round(mean(c.get("pop", 0.0) for c in comps), 4),
        # Top-1 is the book a user sees first — the single most important slot.
        "top1_available":  1 if comps[0].get("avail", 0.0) == 1.0 else 0,
        "top_ids":         [str(b.get("_id", "")) for b in top],
    }


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return round(len(sa & sb) / len(sa | sb), 4)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "reranking_ablation.csv"))
    ap.add_argument("--json", default=str(REPO_ROOT / "experiments" / "reranking_ablation.json"))
    args = ap.parse_args()

    client = MongoClient(os.environ["DB_URL"])
    db = client[os.environ.get("DB_NAME", "test")]
    # Pull full docs once — the ablation re-uses the same candidate pool
    # across configs, eliminating embedding-lookup noise.
    print("Loading books with vectors from MongoDB...")
    books_all = list(db["books"].find({"vector": {"$exists": True}}))
    print(f"  {len(books_all)} books with embeddings available.\n")

    per_query_rows: list[dict] = []
    print(f"Running {len(CONFIGS)} configs × {len(RETRIEVAL_QUERIES)} queries...\n")

    # Cache candidate pools: semantic search is identical across configs,
    # so we run it once per query and reuse the results.
    pools: dict[str, list[dict]] = {}
    for q in RETRIEVAL_QUERIES:
        pools[q] = search_books(q, books_all, top_k=CAND_POOL)

    # For Jaccard overlap vs the proposed config, we need C4's top-k per
    # query before computing differences for C1/C2/C3/C5.
    proposed_id = "C4"
    proposed_top_per_query: dict[str, list[str]] = {}
    for q, pool in pools.items():
        row = measure(dict(CONFIGS[3][2]), pool)   # C4 lives at index 3
        proposed_top_per_query[q] = row["top_ids"]

    # Now measure every config against the cached pool.
    for cfg_id, label, weights in CONFIGS:
        for q in RETRIEVAL_QUERIES:
            row = measure(weights, pools[q])
            row["config_id"]   = cfg_id
            row["config_label"]= label
            row["query"]       = q
            row["jaccard_vs_proposed"] = jaccard(row["top_ids"], proposed_top_per_query[q])
            per_query_rows.append(row)

    # Aggregate per config.
    summary: list[dict] = []
    for cfg_id, label, weights in CONFIGS:
        rows = [r for r in per_query_rows if r["config_id"] == cfg_id]
        summary.append({
            "config_id":            cfg_id,
            "label":                label,
            "alpha":                weights["alpha"],
            "beta":                 weights["beta"],
            "gamma":                weights["gamma"],
            "delta":                weights["delta"],
            "epsilon":              weights["epsilon"],
            "mean_sim":             round(mean(r["mean_sim"] for r in rows), 4),
            "avail_at_k":           round(mean(r["avail_at_k"] for r in rows), 4),
            "mean_rating":          round(mean(r["mean_rating"] for r in rows), 4),
            "mean_pop":             round(mean(r["mean_pop"] for r in rows), 4),
            "top1_available_rate":  round(mean(r["top1_available"] for r in rows), 4),
            "jaccard_vs_proposed":  round(mean(r["jaccard_vs_proposed"] for r in rows), 4),
            "n_queries":            len(rows),
        })

    # Write CSV (thesis table) + JSON (raw data for any later re-analysis).
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "top_k":       TOP_K,
            "cand_pool":   CAND_POOL,
            "configs":     [{"id": c[0], "label": c[1], "weights": c[2]} for c in CONFIGS],
            "per_query":   per_query_rows,
            "summary":     summary,
        }, f, indent=2, default=str)

    # Pretty-print summary for terminal reading.
    print(f"{'ID':<4} {'Label':<22} {'sim':>6} {'avail@k':>8} {'rating':>7} "
          f"{'pop':>6} {'top1_av':>8} {'J↔C4':>7}")
    print("-" * 74)
    for row in summary:
        print(f"{row['config_id']:<4} {row['label']:<22} "
              f"{row['mean_sim']:>6} {row['avail_at_k']:>8} "
              f"{row['mean_rating']:>7} {row['mean_pop']:>6} "
              f"{row['top1_available_rate']:>8} {row['jaccard_vs_proposed']:>7}")

    print(f"\nWrote {args.out} ({len(summary)} rows)")
    print(f"Wrote {args.json} ({len(per_query_rows)} per-query rows)")


if __name__ == "__main__":
    main()

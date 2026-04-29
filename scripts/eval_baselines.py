"""
Three-way baseline comparison — thesis Chapter 5 Table 2.

Research question (RQ1): Does a ReAct-based tool-augmented agent outperform
static pipelines on task completion and hallucination rate?

The experiment answers RQ1 by running THREE systems on the same 35-query
test bank and comparing their 5 metrics:

  B1 — Keyword search
        MongoDB substring match on title + author + description. No
        embeddings, no reasoning, no availability filter. Simulates the
        "before AI" baseline: what a naive campus book app looks like.

  B2 — Semantic retrieval only
        search_books() (cosine similarity on all-MiniLM-L6-v2 embeddings).
        No agent, no re-ranking, no availability check — just vector top-k.
        Isolates the contribution of the dense retrieval layer.

  Proposed — Full ReAct agent (v4 prompt)
        run_react_loop with the hybrid prompt, all 6 tools, the 5-component
        re-ranking formula. Everything the thesis builds.

The delta B1 → B2 shows the contribution of semantic retrieval.
The delta B2 → Proposed shows the contribution of the agent + re-ranking +
hybrid prompt combined (the thesis's architectural work).

Usage
-----
    # Run full suite (all 3 systems × 35 queries). Takes ~15 min on free tier.
    python scripts/eval_baselines.py

    # Single system only (for partial reruns):
    python scripts/eval_baselines.py --systems B1
    python scripts/eval_baselines.py --systems B2,proposed

    # Custom output path:
    python scripts/eval_baselines.py --out experiments/baseline_run3.csv

Environment: needs DB_URL, GROQ_API_KEY. Run 3 times (different days) for
mean ± std dev in the thesis table.
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
import re
import time
from datetime import datetime, timezone

from pymongo import MongoClient
from dotenv import load_dotenv

from app.search import search_books
from app.metrics import score_response, aggregate
from app.eval_queries import ALL_QUERIES, category_of

load_dotenv()

TOP_K = 5


# =========================================================================
# BASELINE 1 — Keyword search
# =========================================================================
# No embeddings. Tokenise the query, find books whose title/author/
# description contains at least one query token, rank by number of token
# matches (higher = more overlap). This is the weakest possible retrieval —
# represents a "naive Mongo app" without any ML.
#
# Notably: this baseline has no concept of "availability" or "ranking by
# popularity" — it returns top-k by crude lexical overlap. That's the
# point; the delta to B2 and Proposed shows what those layers add.
# =========================================================================

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "of", "to", "for", "in", "on",
    "and", "or", "me", "i", "my", "any", "some", "what", "find",
    "show", "get", "need", "want", "do", "you", "have",
})


def b1_keyword_search(query: str, books: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Token-overlap ranking across title + author + description."""
    tokens = [t for t in re.findall(r"[a-zA-Z]{3,}", query.lower()) if t not in STOPWORDS]
    if not tokens:
        return []

    scored = []
    for b in books:
        haystack = " ".join([
            str(b.get("title", "")),
            str(b.get("author", "")),
            str(b.get("description", "")),
            str(b.get("genre", "")),
        ]).lower()
        matches = sum(1 for t in tokens if t in haystack)
        if matches > 0:
            scored.append((matches, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored[:top_k]]


def b1_format_response(query: str, results: list[dict]) -> str:
    """Flatten the keyword search results into the same natural-language
    shape the agent produces — so metrics (keyword overlap with response,
    hallucination-in-response) are computed on directly comparable text."""
    if not results:
        return ("I couldn't find any books matching that. Try a different "
                "search term!")
    lines = ["Here are some books that match your search:"]
    for b in results:
        title   = b.get("title", "Unknown")
        author  = b.get("author", "Unknown")
        avail   = "Available" if b.get("available", True) else "Currently borrowed"
        lines.append(f"- {title} by {author} ({avail})")
    return "\n".join(lines)


# =========================================================================
# BASELINE 2 — Semantic retrieval only
# =========================================================================
# Uses search_books() — the same embedding model and cosine similarity
# the full system uses at its retrieval step — but stops there. No agent,
# no re-ranking, no availability filter. Top-k by raw cosine.
#
# Purpose: isolates what dense retrieval alone adds over keyword matching.
# Also isolates what everything else (agent, re-ranking, prompting) adds
# on top of plain semantic search.
# =========================================================================

def b2_semantic_only(query: str, books: list[dict], top_k: int = TOP_K) -> list[dict]:
    return search_books(query, books, top_k=top_k)


def b2_format_response(query: str, results: list[dict]) -> str:
    if not results:
        return "No books found for that query."
    lines = ["Here are some books related to your query:"]
    for b in results:
        title   = b.get("title", "Unknown")
        author  = b.get("author", "Unknown")
        avail   = "Available" if b.get("available", True) else "Currently borrowed"
        lines.append(f"- {title} by {author} ({avail})")
    return "\n".join(lines)


# =========================================================================
# PROPOSED — Full ReAct agent (v4 prompt)
# =========================================================================
# Re-uses the exact run_react_loop + tool dispatch that production /agent
# endpoint uses. Keeping this as a thin wrapper ensures the baseline and
# production paths run identical code — the comparison is apples-to-apples.
# =========================================================================

def run_proposed(query: str) -> tuple[str, list[str], int]:
    """Lazy-import routes so the Groq client is only initialised when
    the proposed system is actually being evaluated."""
    from app.routes import run_react_loop, tools as tool_defs
    from app.prompts import get_agent_prompt
    messages = [
        {"role": "system", "content": get_agent_prompt("v4")},
        {"role": "user",   "content": query},
    ]
    return run_react_loop(messages, tool_defs, max_iterations=5)


# =========================================================================
# Main experiment loop
# =========================================================================

def run_system(system_id: str, queries: list[str], books: list[dict],
               db_titles: list[str]) -> list[dict]:
    """Run one system across the full query bank. Returns per-query rows
    with metric flags already computed."""
    print(f"\n=== {system_id} ===")
    rows = []
    for i, q in enumerate(queries, 1):
        t0 = time.time()
        try:
            if system_id == "B1":
                results = b1_keyword_search(q, books, top_k=TOP_K)
                response = b1_format_response(q, results)
                tools_called: list[str] = []
                iterations = 0
            elif system_id == "B2":
                results = b2_semantic_only(q, books, top_k=TOP_K)
                response = b2_format_response(q, results)
                tools_called = []
                iterations = 0
            elif system_id == "proposed":
                response, tools_called, iterations = run_proposed(q)
            else:
                raise ValueError(f"unknown system: {system_id}")
            error = None
        except Exception as exc:
            # Log but don't crash the whole run — a single failure shouldn't
            # invalidate the other 34 queries. Marked as failed completion.
            response, tools_called, iterations = "", [], 0
            error = str(exc)
            print(f"  [{i}/{len(queries)}] FAILED: {exc}")

        metrics = score_response(q, response, tools_called, db_titles)
        rows.append({
            "system":        system_id,
            "query":         q,
            "category":      category_of(q),
            "response":      response,
            "tools_called":  tools_called,
            "iterations":    iterations,
            "latency_sec":   round(time.time() - t0, 2),
            "error":         error,
            **metrics,
        })
        print(f"  [{i}/{len(queries)}] {q[:50]:<50} "
              f"{'✓' if metrics['task_complete'] else '✗'} "
              f"{len(tools_called)}t {iterations}i")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="B1,B2,proposed",
                    help="comma-separated subset; default all three")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "baselines.csv"))
    ap.add_argument("--json", default=str(REPO_ROOT / "experiments" / "baselines.json"))
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    queries = ALL_QUERIES

    client = MongoClient(os.environ["DB_URL"])
    db = client[os.environ.get("DB_NAME", "test")]
    print("Loading books from MongoDB...")
    books_all = list(db["books"].find({}))
    books_with_vec = [b for b in books_all if "vector" in b]
    db_titles = [b.get("title", "") for b in books_all]
    print(f"  {len(books_all)} books total ({len(books_with_vec)} with embeddings)")

    all_rows: list[dict] = []
    for sys_id in systems:
        # B2 needs vectors; B1 and proposed use the full set (proposed filters
        # internally when the LLM calls semantic_search).
        corpus = books_with_vec if sys_id == "B2" else books_all
        all_rows.extend(run_system(sys_id, queries, corpus, db_titles))

    # Per-system aggregates for the thesis table.
    summary: list[dict] = []
    for sys_id in systems:
        rows = [r for r in all_rows if r["system"] == sys_id]
        agg = aggregate(rows)
        avg_latency = round(sum(r["latency_sec"] for r in rows) / max(len(rows), 1), 3)
        summary.append({
            "system":                sys_id,
            "n":                     agg["n"],
            "task_completion_rate":  agg["task_completion_rate"],
            "hallucination_rate":    agg["hallucination_rate"],
            "response_relevance":    agg["response_relevance"],
            "tool_precision":        agg["tool_precision"],
            "prompt_adherence_rate": agg["prompt_adherence_rate"],
            "avg_latency_sec":       avg_latency,
        })

    # Write outputs.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Full per-query JSON for any post-hoc re-analysis (per-category
    # breakdowns, error inspection, etc.) without re-running the agent.
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "systems":    systems,
            "n_queries":  len(queries),
            "per_query":  all_rows,
            "summary":    summary,
        }, f, indent=2, default=str)

    # Pretty-print table for the terminal.
    print("\n" + "=" * 78)
    print(f"{'System':<10} {'n':>3} {'TCR':>6} {'HR':>6} {'RR':>6} {'TP':>6} {'PAR':>6} {'lat(s)':>8}")
    print("-" * 78)
    for row in summary:
        print(f"{row['system']:<10} {row['n']:>3} "
              f"{row['task_completion_rate']:>6} {row['hallucination_rate']:>6} "
              f"{row['response_relevance']:>6} {row['tool_precision']:>6} "
              f"{row['prompt_adherence_rate']:>6} {row['avg_latency_sec']:>8}")
    print("=" * 78)
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.json} ({len(all_rows)} per-query rows)")


if __name__ == "__main__":
    main()

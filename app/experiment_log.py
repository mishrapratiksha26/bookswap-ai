"""
experiment_log.py — append-only run log for thesis Chapter 5.

Every curriculum parse and every agent response is logged as one JSON line.
jsonl is chosen over a Mongo collection so the thesis can:
  - Ship the raw logs in the appendix (small, text-based, diffable)
  - Read them straight into pandas with `pd.read_json(path, lines=True)`
  - Avoid needing a DB connection when building result tables offline

File layout:
  bookswap-ai/experiments/curriculum_runs.jsonl
  bookswap-ai/experiments/agent_runs.jsonl

Each row is a self-contained record. No foreign keys. No compaction.
If a run crashes mid-way, the partial record is simply not written —
jsonl never carries half-written rows because we write + flush per call.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# experiments/ lives at the repo root, next to app/ and main.py.
# We resolve it relative to this file so cwd never matters.
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

CURRICULUM_LOG = EXPERIMENTS_DIR / "curriculum_runs.jsonl"
AGENT_LOG      = EXPERIMENTS_DIR / "agent_runs.jsonl"


def _append(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON line to a log file. Never raises — logging a run
    must not break the request that produced it."""
    try:
        # default=str handles ObjectId, datetime, and anything Mongo-ish
        line = json.dumps(record, ensure_ascii=False, default=str)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        print(f"[experiment_log] failed to write to {path}: {e}", flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_curriculum_run(
    *,
    prompt_version: str,
    pdf_filename: str | None,
    raw_text_len: int,
    parsed: dict,
    recommended_books: list[dict],
    unit_matches: list[dict],
    error: str | None = None,
) -> None:
    """
    Called from process_curriculum_pdf at the end of a parse.

    What we store (and why):
      prompt_version      — which prompt template was used
      pdf_filename        — reproduce by re-uploading the same file
      raw_text_len        — tiny sanity check ("did PyMuPDF actually read text?")
      parsed              — raw LLM output (course + units + textbooks)
      recommended_books   — which prof-recommended books matched inventory
      unit_matches        — unit → book → chapter mapping
      match_score_avg     — derived: average unit match score (0..1), useful
                            for the Chapter 5 Curriculum Coverage Score table
      error               — non-null if parse failed (helps triage in results)
    """
    scores = [u.get("match_score", 0.0) for u in (unit_matches or [])]
    match_score_avg = round(sum(scores) / len(scores), 4) if scores else 0.0

    _append(CURRICULUM_LOG, {
        "timestamp":         _now(),
        "prompt_version":    prompt_version,
        "pdf_filename":      pdf_filename,
        "raw_text_len":      raw_text_len,
        "course_name":       parsed.get("course_name", "") if parsed else "",
        "course_code":       parsed.get("course_code", "") if parsed else "",
        "department":        parsed.get("department", "")  if parsed else "",
        "n_units_parsed":    len(parsed.get("units", []))        if parsed else 0,
        "n_textbooks":       len(parsed.get("textbooks", []))    if parsed else 0,
        "n_references":      len(parsed.get("reference_books", [])) if parsed else 0,
        "n_books_found":     sum(1 for r in (recommended_books or []) if r.get("found")),
        "n_books_missing":   sum(1 for r in (recommended_books or []) if not r.get("found")),
        "n_unit_matches":    len(unit_matches or []),
        "match_score_avg":   match_score_avg,
        "recommended_books": recommended_books,
        "unit_matches":      unit_matches,
        "error":             error,
    })


def log_agent_run(
    *,
    prompt_version: str,
    user_id: str | None,
    session_id: str | None,
    query: str,
    response: str,
    tools_called: list[str],
    iterations: int,
    error: str | None = None,
) -> None:
    """
    Called from /agent at the end of every ReAct loop run.

    Why log tools_called + iterations?
      - tools_called is the input to Tool Precision (metric 1 in thesis)
      - iterations shows how many loop cycles the prompt required — a proxy
        for whether CoT scaffolding reduced back-and-forth
      - Response text is stored verbatim so Hallucination Rate and Response
        Relevance can be recomputed offline without re-hitting Groq.
    """
    _append(AGENT_LOG, {
        "timestamp":        _now(),
        "prompt_version":   prompt_version,
        "user_id":          user_id,
        "session_id":       session_id,
        "query":            query,
        "response":         response,
        "tools_called":     tools_called,
        "n_tools_called":   len(tools_called or []),
        "iterations":       iterations,
        "error":            error,
    })

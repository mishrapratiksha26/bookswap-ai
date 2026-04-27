"""
Evaluation metrics for BookSwap thesis Chapter 5.

The 5 core metrics (Tool Precision, Hallucination Rate, Response Relevance,
Task Completion, Prompt Adherence) were first implemented inline inside the
POST /evaluate handler in routes.py. As the ablation studies grew — prompt
ablation, re-ranking ablation, three-way baseline, curriculum coverage —
having the logic duplicated in four scripts became painful: a fix in one
place would silently diverge from the others.

This module is the single source of truth. /evaluate and every script in
scripts/eval_* import from here. No functional change from the original
inline implementation — only extracted and lightly refactored so each
metric is an independent pure function with a doc-commented contract.

Design notes
  - Every function takes its inputs explicitly (no global DB reads). That
    makes the metrics unit-testable and lets offline scripts compute them
    without hitting MongoDB.
  - `score_response()` is the convenience wrapper that runs all 5 at once
    and returns a dict — used by /evaluate to keep response shape stable.
  - Hallucination Rate is deliberately kept as the "conservative" proxy
    originally used (mentions DB titles but called no search tool). A
    stricter version that parses bold-title markdown against DB lookups
    is possible later — noted in thesis further scope.
"""
from __future__ import annotations
from typing import Iterable, Sequence

# Forbidden tokens — if these appear in the user-facing response, the agent
# has leaked internal state. Kept identical to the original /evaluate list
# so v1-v4 prompt comparison numbers stay comparable with earlier runs.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    '"_id"', '"book_id"', '"tool_call"', 'function_name',
    'ObjectId', '"score":', '"vector"',
)

# Keywords that indicate a book-retrieval query. Used by tool_precision to
# decide whether a search-tool call was appropriate vs off-topic noise.
BOOK_QUERY_KEYWORDS: tuple[str, ...] = (
    "book", "borrow", "find", "suggest", "recommend", "pdf", "notes",
    "thriller", "available", "fiction", "textbook", "previous year", "papers",
    "study", "syllabus", "chapter",
)

# Tool names counted as "appropriate" for a book query.
RETRIEVAL_TOOLS: frozenset[str] = frozenset({
    "semantic_search", "check_availability", "get_alternatives",
    "search_pdfs", "get_user_profile",
})


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def task_completion(response: str) -> bool:
    """
    Task Completion: response is non-empty, not an apology stub, long enough
    to plausibly carry useful content. This is a proxy for "did the agent
    actually produce an answer?" — distinguishes from rate-limit / timeout
    fallbacks that start "Sorry".
    """
    if not response:
        return False
    if response.startswith("Sorry"):
        return False
    return len(response) > 20


def hallucination(
    response: str,
    tools_called: Sequence[str],
    db_titles: Iterable[str],
) -> bool:
    """
    Hallucination flag: True if the response mentions a book title that
    exists in our DB BUT the agent never called a search tool to retrieve
    it. This catches the common failure mode where the LLM recites a book
    from training data that happens to be in our library — the title is
    "real" but the agent didn't actually retrieve it, so the recommendation
    is not grounded in tool output.

    Conservative proxy. A stricter version would parse markdown bold-titles
    from the response and check each against the DB; this version flags a
    whole response. Kept identical to the original /evaluate implementation
    so results across experiments remain comparable.
    """
    search_tools = [t for t in tools_called if "search" in t]
    if search_tools:
        return False
    lc_response = response.lower()
    return any(
        title and title in lc_response
        for title in (t.lower() for t in db_titles)
    )


def response_relevance(query: str, response: str) -> float:
    """
    Response Relevance: Jaccard-style keyword overlap between query and
    response, normalised by query length. Reference-free — no human
    annotation needed. Reports how much of what the user asked about
    actually appears in the answer.

    Returns: [0.0, 1.0].
    """
    q_words = set(query.lower().split())
    r_words = set(response.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & r_words) / len(q_words)


def tool_precision_per_query(query: str, tools_called: Sequence[str]) -> tuple[int, int]:
    """
    Tool Precision (per-query raw counts).

    Returns: (appropriate_calls, total_calls_including_zero_baseline).

    Rules:
      - Book query    → each retrieval-tool call counts as appropriate.
      - Off-topic     → calling zero tools is the correct behaviour; we
                        count that as 1 appropriate / 1 total so off-topic
                        restraint contributes positively to the aggregate.
      - Off-topic + tool calls → each call is inappropriate (denominator
                        grows, numerator unchanged).

    Aggregate across queries: sum numerator / sum denominator.
    """
    is_book_query = any(kw in query.lower() for kw in BOOK_QUERY_KEYWORDS)

    if not tools_called:
        # Restraint case: right for off-topic, wrong for book queries.
        return (0 if is_book_query else 1, 1)

    if is_book_query:
        approp = sum(1 for t in tools_called if t in RETRIEVAL_TOOLS)
        return (approp, len(tools_called))
    else:
        # Off-topic but tools were called → none of those calls are
        # appropriate. Denominator = number of calls.
        return (0, len(tools_called))


def prompt_adherence(response: str) -> bool:
    """
    Prompt Adherence: response contains no forbidden internal-state leaks
    (raw IDs, tool-call JSON, MongoDB debris). True = adhered.
    """
    return not any(fp in response for fp in FORBIDDEN_PATTERNS)


# ---------------------------------------------------------------------------
# Aggregate wrapper
# ---------------------------------------------------------------------------

def score_response(
    query: str,
    response: str,
    tools_called: Sequence[str],
    db_titles: Iterable[str],
) -> dict:
    """
    Compute all five metrics for a single (query, response, tools) triple.
    Returns per-query dict. The caller aggregates across queries.

    Why per-query and not pre-aggregated: keeps raw data available for
    thesis Chapter 5 error-bar calculation (mean ± std dev across the 35
    queries) without re-running the agent.
    """
    approp, total = tool_precision_per_query(query, tools_called)
    return {
        "task_complete":   task_completion(response),
        "hallucinated":    hallucination(response, tools_called, db_titles),
        "relevance":       response_relevance(query, response),
        "tp_numerator":    approp,
        "tp_denominator":  total,
        "prompt_adherent": prompt_adherence(response),
    }


def aggregate(per_query_rows: list[dict]) -> dict:
    """
    Aggregate a list of per-query metric dicts (output of score_response)
    into the 5 headline rates used in the thesis tables.

    Returns numbers in [0, 1] — not percentages. Table rendering handles
    the ×100 if needed.
    """
    n = max(len(per_query_rows), 1)
    tp_num = sum(r["tp_numerator"]   for r in per_query_rows)
    tp_den = max(sum(r["tp_denominator"] for r in per_query_rows), 1)
    return {
        "n":                      len(per_query_rows),
        "task_completion_rate":   round(sum(1 for r in per_query_rows if r["task_complete"]) / n, 4),
        "hallucination_rate":     round(sum(1 for r in per_query_rows if r["hallucinated"]) / n, 4),
        "response_relevance":     round(sum(r["relevance"] for r in per_query_rows) / n, 4),
        "tool_precision":         round(tp_num / tp_den, 4),
        "prompt_adherence_rate":  round(sum(1 for r in per_query_rows if r["prompt_adherent"]) / n, 4),
    }

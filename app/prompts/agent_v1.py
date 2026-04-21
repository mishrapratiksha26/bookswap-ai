"""
Agent System Prompt — v1 (baseline)

Thesis role: performance floor. Establishes what a naive, rule-less prompt
produces so later versions have something to improve over.

Techniques used:
  (none — this is intentionally minimal)

Expected failures when run on the 35-query bank:
  - Hallucinated book titles (LLM falls back to training-data books)
  - Tool-call skipping (no rule forces semantic_search before answering)
  - JSON/ID leaks (no rule forbids them)
  - No clarifying question on vague queries
"""

PROMPT = "You are a helpful assistant. Help users find books."

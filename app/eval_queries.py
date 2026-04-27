"""
Canonical 35-query test bank for all thesis Chapter 5 experiments.

Extracted from the original inline list inside /evaluate so every ablation
(prompt version, re-ranking weights, three-way baseline) runs on the exact
same inputs. If this list changes, every previous experiment's numbers
become non-comparable — so treat it as frozen.

Structure:
  35 queries, 5 categories × 7 each.

  cat_1_direct_search  — unambiguous book lookups. Measures baseline retrieval.
  cat_2_availability   — queries that exercise the check_availability + alternatives path.
  cat_3_vague          — ambiguous queries. Tests clarification behaviour (prompt-dependent).
  cat_4_offtopic       — out-of-scope. Tests restraint (should call 0 tools).
  cat_5_pdf_material   — study-material queries. Exercises the search_pdfs path.
"""

CATEGORIES: dict[str, list[str]] = {
    "cat_1_direct_search": [
        "Find me thriller books",
        "I need books on data structures",
        "Do you have any self-help books?",
        "Show me books by Colleen Hoover",
        "Find books about leadership",
        "I want to read a mystery novel",
        "Any motivational books available?",
    ],
    "cat_2_availability": [
        "Is Verity available to borrow?",
        "Can I borrow The Housemaid?",
        "Find me something like The Silent Patient",
        "What thriller books can I borrow right now?",
        "Show me available psychology books",
        "I want to borrow a fiction book this week",
        "Any books available on leadership?",
    ],
    "cat_3_vague": [
        "Suggest me a good book",
        "What are some easy reads?",
        "I need something interesting",
        "Recommend me something",
        "Any new arrivals?",
        "What's popular right now?",
        "I'm bored, help me find a book",
    ],
    "cat_4_offtopic": [
        "What is the weather today?",
        "Help me write a Python script",
        "Who is the Prime Minister of India?",
        "Can you book me a flight?",
        "What is machine learning?",
        "Tell me a joke",
        "Help me with my math homework",
    ],
    "cat_5_pdf_material": [
        "Find OS notes for CSE",
        "Any DBMS previous year papers?",
        "Show me reference material for algorithms",
        "I need study material for networks",
        "Find textbooks for discrete mathematics",
        "Any CSE department notes?",
        "Previous year papers for data structures",
    ],
}

# Flat ordered list — identical ordering to the original /evaluate handler
# so previously logged runs (experiments/agent_runs.jsonl) remain aligned.
ALL_QUERIES: list[str] = [
    q for cat in CATEGORIES.values() for q in cat
]

# Retrieval-only subset — used by the re-ranking ablation and three-way
# baseline since vague / off-topic queries don't produce a ranked book list
# whose re-ranking can be meaningfully compared across weight configs.
RETRIEVAL_QUERIES: list[str] = (
    CATEGORIES["cat_1_direct_search"]
    + CATEGORIES["cat_2_availability"]
    + CATEGORIES["cat_5_pdf_material"]
)


def category_of(query: str) -> str:
    """Reverse-lookup which category a query belongs to (for per-category stats)."""
    for cat_name, queries in CATEGORIES.items():
        if query in queries:
            return cat_name
    return "unknown"

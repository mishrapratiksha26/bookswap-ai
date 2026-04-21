# Experiments — thesis Chapter 5 raw data

This directory holds append-only JSON-Lines logs of every curriculum parse
and every agent run, keyed by prompt version. They are the raw data behind
the results tables in Chapter 5.

## Files

| File | One row per | Use in thesis |
|---|---|---|
| `curriculum_runs.jsonl` | lecture-plan upload | Curriculum Coverage Score (CCS), Avg Resources Per Topic (ARPT), v1→v2 parse-quality delta |
| `agent_runs.jsonl`      | `/agent` response   | Tool Precision, Hallucination Rate, Response Relevance, Task Completion, Prompt Adherence across prompt versions v1→v4 |

## Reading into pandas

```python
import pandas as pd
cur = pd.read_json("experiments/curriculum_runs.jsonl", lines=True)
agn = pd.read_json("experiments/agent_runs.jsonl",      lines=True)

# Example: prompt-version ablation (Table 1 in Chapter 5)
agn.groupby("prompt_version")[["n_tools_called", "iterations"]].mean()
```

## Row schema — `curriculum_runs.jsonl`

```
timestamp          ISO-8601 UTC
prompt_version     "v1" | "v2"  — which curriculum prompt ran
pdf_filename       original upload filename (for reproduction)
raw_text_len       characters extracted by PyMuPDF from the PDF
course_name        parsed by LLM
course_code        parsed by LLM
department         parsed by LLM (after DEPARTMENT RULE fix in v2)
n_units_parsed     LLM-extracted unit count
n_textbooks        LLM-extracted textbook count
n_references       LLM-extracted reference-book count
n_books_found      professor books that matched inventory via vector search
n_books_missing    professor books NOT found in inventory
n_unit_matches     units successfully mapped to a book + chapter
match_score_avg    mean cosine sim across unit matches (0..1)
recommended_books  full array — the book cards rendered on the result page
unit_matches       full array — the unit→chapter map rendered on the result page
error              non-null if parse failed
```

## Row schema — `agent_runs.jsonl`

```
timestamp          ISO-8601 UTC
prompt_version     "v1" | "v2" | "v3" | "v4"
user_id            requester (null for guests)
session_id         conversation id
query              user's message
response           final assistant message
tools_called       ordered list of tool names
n_tools_called     convenience count
iterations         number of ReAct loop cycles taken
error              non-null if the loop failed
```

## Git policy

Both `.jsonl` files are `.gitignore`d — they grow with every request and
aren't source. Commit specific frozen snapshots into `experiments/snapshots/`
(e.g. `2026-04-21_v4_35queries.jsonl`) when you freeze a result for the
thesis; those snapshots should be committed so the paper is reproducible.

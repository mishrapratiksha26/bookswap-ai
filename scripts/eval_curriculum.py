"""
Curriculum Coverage evaluation — thesis Chapter 5, novel-metric table.

Research question (RQ3): Does curriculum-aware retrieval with chapter-level
matching improve resource coverage compared to title-level matching?

This is the thesis's domain-specific contribution — two novel metrics not
borrowed from existing agent / retrieval literature:

  CCS  (Curriculum Coverage Score)
    = |units with ≥1 matched resource| / |total units in the plan|
    Interpretable as: "what fraction of this course's syllabus can a
    student actually find material for on BookSwap?"

  ARPT (Average Resources Per Topic)
    = Σ matches across units / |total units|
    A richness measure — higher means a student has more options per unit.

  PBR  (Professor Books found Rate)
    = |textbooks found in inventory| / |textbooks the professor recommended|
    Honest view: how often does the library actually have what the
    instructor asked for? Useful complement to CCS — the professor's
    textbook may not cover every unit, but it's the "ground truth"
    recommendation whose presence is a strong positive signal.

Inputs
------
Expects lecture-plan PDFs in experiments/sample_lecture_plans/. Drop in
2–5 real IIT ISM lecture plans (any department) before running. Each
PDF is run through the full pipeline — parse_lecture_plan (Groq), then
find_recommended_books_in_inventory + map_units_to_chapters against the
live MongoDB inventory.

Usage
-----
    # Put lecture plan PDFs here first:
    #   experiments/sample_lecture_plans/MCC510_OS.pdf
    #   experiments/sample_lecture_plans/MCO502_Optimization.pdf

    python scripts/eval_curriculum.py
    python scripts/eval_curriculum.py --pdf-dir custom/path
    python scripts/eval_curriculum.py --prompt-version v2

Environment: DB_URL, GROQ_API_KEY. One Groq call per PDF (parse).
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
from datetime import datetime, timezone
from statistics import mean

from pymongo import MongoClient
from dotenv import load_dotenv

from app.chapter_extractor import process_curriculum_pdf

load_dotenv()

DEFAULT_PDF_DIR = REPO_ROOT / "experiments" / "sample_lecture_plans"


def compute_metrics(result: dict) -> dict:
    """Given the dict returned by process_curriculum_pdf, compute CCS,
    ARPT, and PBR plus supporting counts."""
    unit_matches = result.get("unit_matches", []) or []
    recommended  = result.get("recommended_books", []) or []

    # CCS: fraction of units with at least one match.
    total_units = len(unit_matches)
    covered     = sum(1 for u in unit_matches if u.get("matches"))
    ccs = round(covered / total_units, 4) if total_units else 0.0

    # ARPT: total matched resources / total units. Counts every entry in
    # each unit's matches list — a unit with 3 matches contributes 3.
    total_matches = sum(len(u.get("matches", []) or []) for u in unit_matches)
    arpt = round(total_matches / total_units, 4) if total_units else 0.0

    # PBR: fraction of professor-recommended textbooks found in inventory.
    # `found` is the boolean flag set by find_recommended_books_in_inventory.
    total_rec = len(recommended)
    found_rec = sum(1 for r in recommended if r.get("found"))
    pbr = round(found_rec / total_rec, 4) if total_rec else 0.0

    # Mean unit-level match score — average of the top match_score per unit
    # (the cosine sim of the best chapter/book hit). Useful as a quality
    # signal to complement the binary covered/not-covered in CCS.
    top_scores = [
        u["matches"][0].get("match_score", 0.0)
        for u in unit_matches if u.get("matches")
    ]
    mean_top_score = round(mean(top_scores), 4) if top_scores else 0.0

    return {
        "course_code":      result.get("course_code", ""),
        "course_name":      result.get("course_name", ""),
        "department":       result.get("department", ""),
        "n_units":          total_units,
        "n_units_covered":  covered,
        "n_total_matches":  total_matches,
        "n_recommended":    total_rec,
        "n_recommended_found": found_rec,
        "ccs":              ccs,
        "arpt":             arpt,
        "pbr":              pbr,
        "mean_top_score":   mean_top_score,
        "error":            result.get("error"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR),
                    help=f"directory with lecture plan PDFs (default: {DEFAULT_PDF_DIR})")
    ap.add_argument("--prompt-version", default=None,
                    help="curriculum prompt version (default: latest)")
    ap.add_argument("--out", default=str(REPO_ROOT / "experiments" / "curriculum_coverage.csv"))
    ap.add_argument("--json", default=str(REPO_ROOT / "experiments" / "curriculum_coverage.json"))
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        pdf_dir.mkdir(parents=True, exist_ok=True)

    pdfs_to_process = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs_to_process:
        print(f"\nNo PDFs found in {pdf_dir}.")
        print("\nPlace 2–5 IIT ISM lecture plan PDFs here, e.g.:")
        print(f"  {pdf_dir / 'MCC510_OS.pdf'}")
        print(f"  {pdf_dir / 'MCO502_Optimization.pdf'}")
        print("\nThen re-run this script.")
        sys.exit(0)

    client = MongoClient(os.environ["DB_URL"])
    db = client[os.environ.get("DB_NAME", "test")]
    print("Loading books + pdfs from MongoDB...")
    books_all = list(db["books"].find({}))
    pdfs_all  = list(db["pdfs"].find({}))
    print(f"  {len(books_all)} books, {len(pdfs_all)} PDFs\n")

    rows: list[dict] = []
    for pdf_path in pdfs_to_process:
        print(f"Processing {pdf_path.name}...")
        pdf_bytes = pdf_path.read_bytes()
        result = process_curriculum_pdf(
            pdf_bytes,
            books=books_all,
            pdfs=pdfs_all,
            prompt_version=args.prompt_version,
            pdf_filename=pdf_path.name,
        )
        metrics = compute_metrics(result)
        metrics["pdf_filename"]   = pdf_path.name
        metrics["prompt_version"] = result.get("prompt_version", args.prompt_version or "default")
        rows.append(metrics)

        if metrics["error"]:
            print(f"  ERROR: {metrics['error']}\n")
        else:
            print(f"  {metrics['course_code']} — {metrics['course_name']}")
            print(f"  CCS={metrics['ccs']}  ARPT={metrics['arpt']}  PBR={metrics['pbr']}")
            print(f"  units={metrics['n_units']} (covered {metrics['n_units_covered']}), "
                  f"prof_books={metrics['n_recommended_found']}/{metrics['n_recommended']}\n")

    # Aggregate across PDFs for the thesis-table summary row.
    valid = [r for r in rows if not r["error"] and r["n_units"] > 0]
    summary = {
        "n_plans_run":          len(rows),
        "n_plans_valid":        len(valid),
        "mean_ccs":             round(mean(r["ccs"]  for r in valid), 4) if valid else None,
        "mean_arpt":            round(mean(r["arpt"] for r in valid), 4) if valid else None,
        "mean_pbr":             round(mean(r["pbr"]  for r in valid), 4) if valid else None,
        "mean_top_score":       round(mean(r["mean_top_score"] for r in valid), 4) if valid else None,
    }

    # Write outputs — CSV for the thesis table, JSON for later drill-down.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdf_filename", "course_code", "course_name", "department",
        "n_units", "n_units_covered", "n_total_matches",
        "n_recommended", "n_recommended_found",
        "ccs", "arpt", "pbr", "mean_top_score",
        "prompt_version", "error",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "pdf_dir":    str(pdf_dir),
            "per_plan":   rows,
            "summary":    summary,
        }, f, indent=2, default=str)

    # Terminal summary.
    print("=" * 64)
    print(f"{'Course':<25} {'units':>6} {'cov':>4} {'CCS':>6} {'ARPT':>6} {'PBR':>6}")
    print("-" * 64)
    for r in rows:
        label = (r["course_code"] or r["pdf_filename"])[:24]
        print(f"{label:<25} {r['n_units']:>6} {r['n_units_covered']:>4} "
              f"{r['ccs']:>6} {r['arpt']:>6} {r['pbr']:>6}")
    print("-" * 64)
    if valid:
        print(f"{'MEAN':<25} {'':>6} {'':>4} "
              f"{summary['mean_ccs']:>6} {summary['mean_arpt']:>6} {summary['mean_pbr']:>6}")
    print("=" * 64)
    print(f"\nWrote {args.out}")
    print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()

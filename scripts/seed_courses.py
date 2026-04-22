"""
seed_courses.py — Seeds the `courses` MongoDB collection from data/courses.json.

Idempotent: upserts on `code` so re-running only touches changed docs.
Run once after fetch_iitism_courses.py produces a fresh courses.json, OR
whenever IIT ISM updates the catalogue (re-run the scraper first, then this).

Usage:
    python scripts/seed_courses.py            # normal upsert
    python scripts/seed_courses.py --wipe     # drop the collection first
                                              # (use only if schema changes)

Reads DB_URL + DB_NAME from environment (same pattern as app/routes.py).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import ASCENDING, MongoClient
from pymongo.errors import BulkWriteError

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
COURSES_JSON = REPO_ROOT / "data" / "courses.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wipe",
        action="store_true",
        help="Drop the courses collection before seeding (destructive).",
    )
    args = parser.parse_args()

    db_url = os.environ.get("DB_URL")
    db_name = os.environ.get("DB_NAME", "books")
    if not db_url:
        print("ERROR: DB_URL not set in environment (.env).", file=sys.stderr)
        sys.exit(1)

    if not COURSES_JSON.exists():
        print(
            f"ERROR: {COURSES_JSON} not found. "
            f"Run `python scripts/fetch_iitism_courses.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(COURSES_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)

    courses = payload.get("courses", [])
    if not courses:
        print("ERROR: courses.json contains 0 courses. Bailing out.", file=sys.stderr)
        sys.exit(1)

    client = MongoClient(db_url)
    db = client[db_name]
    col = db["courses"]

    if args.wipe:
        col.drop()
        print(f"Dropped collection: {db_name}.courses")

    # Indexes:
    #   code (unique)       — canonical primary key (e.g. MCC510)
    #   department_code     — fast filter for the "courses in MC" dropdown
    #   name (text)         — substring / text search by course name
    col.create_index([("code", ASCENDING)], unique=True, name="uniq_code")
    col.create_index([("department_code", ASCENDING)], name="by_dept")
    try:
        col.create_index([("name", "text")], name="text_name")
    except Exception:
        pass  # text index already exists — safe to ignore

    # Bulk upsert keyed on `code`. Using update_one with upsert=True per course
    # for clarity (1570 rows, one-time-ish operation — no need for bulk_write
    # complexity). Shape stored in DB matches the JSON 1-to-1 so downstream
    # readers don't have to transform.
    inserted = 0
    modified = 0
    for course in courses:
        code = course.get("code")
        if not code:
            continue
        result = col.update_one(
            {"code": code},
            {"$set": course},
            upsert=True,
        )
        if result.upserted_id is not None:
            inserted += 1
        elif result.modified_count:
            modified += 1

    total_in_db = col.count_documents({})

    print("Seed complete.")
    print(f"  Source file         : {COURSES_JSON}")
    print(f"  Source count        : {len(courses)}")
    print(f"  Inserted (new)      : {inserted}")
    print(f"  Updated (changed)   : {modified}")
    print(f"  Unchanged (skipped) : {len(courses) - inserted - modified}")
    print(f"  Total in DB         : {total_in_db}")
    print(f"  Collection          : {db_name}.courses")
    print()
    print("Sample query — `db.courses.find({department_code:'MC'}).limit(3)`:")
    for doc in col.find({"department_code": "MC"}).limit(3):
        print(f"  {doc['code']:>8}  {doc['name']}")


if __name__ == "__main__":
    try:
        main()
    except BulkWriteError as exc:
        print(f"BulkWriteError: {exc.details}", file=sys.stderr)
        sys.exit(2)

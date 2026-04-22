"""
fetch_iitism_courses.py — One-shot scraper for IIT (ISM) Dhanbad course catalogues.

Downloads the NEP catalogue PDF for every department from people.iitism.ac.in,
parses them with PyMuPDF, and produces a single courses.json with
{code, name, department, department_code} for every course in the institute.

This feeds:
  - Course dropdown on PDF upload form (views/pdfs/new.ejs)
  - Course dropdown on Book upload form (views/books/new.ejs) when genre=EDUCATIONAL
  - The course-strict filter in find_course_related_notes()

Run once (or whenever IIT ISM updates the catalogue):
    python scripts/fetch_iitism_courses.py
"""

import json
import re
import urllib.request
from pathlib import Path

import fitz  # PyMuPDF

# Canonical (department name, department code, catalogue PDF URL)
DEPARTMENTS = [
    ("Applied Geology", "AGL", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/agl/agl.pdf"),
    ("Applied Geophysics", "AGP", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/agp/agp.pdf"),
    ("Chemical Engineering", "CHE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/che/che.pdf"),
    ("Chemistry and Chemical Biology", "CCB", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/ccb/ccb.pdf"),
    ("Civil Engineering", "CVE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/cve/cve.pdf"),
    ("Computer Science and Engineering", "CSE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/cse/cse_new.pdf"),
    ("Electrical Engineering", "EE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/ee/ee.pdf"),
    ("Electronics Engineering", "ECE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/ece/ece.pdf"),
    ("Environmental Science & Engineering", "ESE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/ese/ese.pdf"),
    ("Fuel Mineral & Metallurgical Engineering", "FMME", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/fmme/fmme.pdf"),
    ("Humanities and Social Sciences", "HSS", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/hss/hss_new.pdf"),
    ("Management Studies and Industrial Engineering", "MSIE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/ms/ms_new.pdf"),
    ("Mathematics and Computing", "MC", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/mc/mc.pdf"),
    ("Mechanical Engineering", "MECH", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/mech/mech.pdf"),
    ("Mining Engineering", "ME", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/me/me_new.pdf"),
    ("Petroleum Engineering", "PE", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/pe/pe.pdf"),
    ("Physics", "PHY", "https://people.iitism.ac.in/~academics/assets/course_structure/new/cat/phy/phy.pdf"),
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "iitism_courses"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "courses.json"

# Course codes on the catalogue PDFs are 3 uppercase letters + 3 digits
# (e.g. MCI101, MCC202, MCO502, CSE301). Some physics/HSS codes use 2 letters.
COURSE_CODE_RE = re.compile(r"^[A-Z]{2,4}\d{3}$")
# L-T-P strings look like "3-0-0", "0-0-2", "2-2-0", sometimes "3-1-0", "0-0-0 (S/X)"
LTP_RE = re.compile(r"^\d-\d-\d(\s*\(.+?\))?$")


def download(url: str, dest: Path) -> bool:
    """Download PDF if not already cached."""
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (BookSwap thesis scraper)"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as fh:
            fh.write(resp.read())
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"  ! download failed: {e}")
        return False


def parse_catalogue(pdf_path: Path, dept_name: str, dept_code: str) -> list[dict]:
    """
    Parse a catalogue PDF and return a list of {code, name, ltp, course_type,
    department, department_code} dicts.

    Strategy: catalogue PDFs lay out each course as 5 consecutive lines:
        <serial number>
        <course code>
        <course name>
        <L-T-P>
        <course type>
    We scan the token stream for a course-code token and use its ±N context
    to recover the name + L-T-P + type. This tolerates broken header rows
    and duplicate courses across elective sections.
    """
    doc = fitz.open(pdf_path)
    lines: list[str] = []
    for page in doc:
        for raw in page.get_text().splitlines():
            stripped = raw.strip()
            if stripped:
                lines.append(stripped)
    doc.close()

    courses: list[dict] = []
    seen_codes: set[str] = set()

    i = 0
    while i < len(lines):
        token = lines[i]
        if COURSE_CODE_RE.match(token):
            # Peek ahead: course name is the next non-header line;
            # L-T-P should appear within the next 3 lines.
            name_idx = i + 1
            ltp_idx = None
            for j in range(i + 2, min(i + 6, len(lines))):
                if LTP_RE.match(lines[j]):
                    ltp_idx = j
                    break
            if ltp_idx is None:
                i += 1
                continue

            # Name can span multiple lines between code and L-T-P
            name = " ".join(lines[name_idx:ltp_idx]).strip()
            ltp = lines[ltp_idx]
            course_type = lines[ltp_idx + 1] if ltp_idx + 1 < len(lines) else ""

            if token not in seen_codes and name and len(name) > 2:
                courses.append({
                    "code": token,
                    "name": name,
                    "ltp": ltp,
                    "course_type": course_type if course_type in {"Theory", "Practical", "Modular", "Audit", "Project", "Seminar"} else "",
                    "department": dept_name,
                    "department_code": dept_code,
                })
                seen_codes.add(token)
            i = ltp_idx + 2
            continue
        i += 1

    return courses


def main():
    all_courses: list[dict] = []
    per_dept_counts: dict[str, int] = {}

    for dept_name, dept_code, url in DEPARTMENTS:
        pdf_path = DATA_DIR / f"{dept_code.lower()}_catalogue.pdf"
        print(f"[{dept_code}] {dept_name}")
        if not download(url, pdf_path):
            print(f"  ! skipping {dept_code}")
            continue

        courses = parse_catalogue(pdf_path, dept_name, dept_code)
        per_dept_counts[dept_code] = len(courses)
        print(f"  {len(courses)} courses parsed")
        all_courses.extend(courses)

    # Write unified JSON
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "source": "IIT (ISM) Dhanbad — NEP 2024-25 course catalogues",
            "scraped_from": "people.iitism.ac.in/~academics/Academic/course_wise_pg",
            "total_courses": len(all_courses),
            "per_department": per_dept_counts,
            "courses": all_courses,
        }, fh, indent=2, ensure_ascii=False)

    print()
    print(f"Wrote {len(all_courses)} courses across {len(per_dept_counts)} departments -> {OUT_PATH}")
    print("Per-department counts:")
    for code, n in sorted(per_dept_counts.items()):
        print(f"  {code:>5}  {n:>4}")


if __name__ == "__main__":
    main()

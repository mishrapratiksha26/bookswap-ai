"""
chapter_extractor.py — Phase 7 (Chapter 3.6 of thesis)

Curriculum-Aware Recommendation Pipeline:
  1. Lecture plan PDF uploaded by professor/student (standardised 2-page IIT ISM format)
  2. PyMuPDF extracts raw text from each page
  3. Groq (Llama 3.3-70B) parses topics into a structured list
  4. Topics are matched against the book inventory using semantic search
  5. For each matched book, chapter-level headings are extracted and ranked
     by cosine similarity to the topic query

This bridges the gap between a professor's syllabus and the library inventory —
students see exactly WHICH CHAPTER of WHICH BOOK covers their current lecture topic.

Mathematical basis (thesis Section 3.6):
  For topic t and book b with chapter headings {h₁, h₂, ..., hₙ}:
    chapter_score(hᵢ) = cosine_similarity(embed(t), embed(hᵢ))
    best_chapter = argmax hᵢ  chapter_score(hᵢ)

Fallback chain:
  PyMuPDF → text extraction → Groq parsing → semantic matching
  If PyMuPDF returns < 50 chars/page (scanned/image PDF): return raw text and let
  Groq do its best, or return a graceful "could not parse" message.
"""

import fitz  # PyMuPDF
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from app.embeddings import generate_embedding
from app.prompts import get_curriculum_prompt, CURRICULUM_LATEST
from app.experiment_log import log_curriculum_run

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ---------------------------------------------------------------------------
# STEP 1 — Extract raw text from a lecture plan PDF
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Use PyMuPDF (fitz) to extract raw text from a PDF file.

    Handles:
      - Digital PDFs: high-quality text extraction
      - Scanned/image PDFs: returns low-char-count warning string

    Returns:
      Extracted text as a single string. Each page separated by '\\n\\n--- PAGE N ---\\n'.
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(f"--- PAGE {page_num} ---\n{text}")
        doc.close()
        full_text = "\n\n".join(pages)
        return full_text
    except Exception as e:
        return f"ERROR: Could not read PDF — {str(e)}"


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Same as extract_text_from_pdf but accepts bytes (for in-memory uploads)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append(f"--- PAGE {page_num} ---\n{text}")
        doc.close()
        return "\n\n".join(pages)
    except Exception as e:
        return f"ERROR: Could not read PDF — {str(e)}"


# ---------------------------------------------------------------------------
# STEP 2 — Parse topics from extracted text using Groq
# ---------------------------------------------------------------------------

def parse_lecture_plan(text: str, prompt_version: str | None = None) -> dict:
    """
    Parse a complete IIT ISM lecture plan PDF text into structured data.

    Args:
      text            raw text extracted from the lecture-plan PDF
      prompt_version  which curriculum prompt template to use ("v1", "v2", ...)
                      defaults to the latest version in app/prompts/__init__.py

    Returns:
      dict with course_name, course_code, department, units[], textbooks[],
      reference_books[]. Empty dict on failure (error is printed to stderr).

    Why one Groq call instead of two?
      The old approach made one call for topics only — ignoring the book list
      on page 2 entirely. The professor's recommended books are ground truth.
      We should check those first before doing any semantic search. One
      structured call extracts everything in one shot.

    Why prompt_version is a first-class argument (thesis Chapter 5):
      The curriculum-parser prompt evolves (v1 → v2 → ...). To produce the
      ablation table in the results chapter we need to run the SAME PDF through
      different prompt versions without patching the source. Pass the version
      in; get back whatever that version's prompt produces.
    """
    if not text or len(text.strip()) < 50:
        return {}

    try:
        prompt_template = get_curriculum_prompt(prompt_version)
        # NOTE: use .replace(), NOT .format(). The prompt contains literal
        # JSON curly braces as an example of the desired output shape, and
        # str.format() would interpret every "{" as a format placeholder
        # and raise ValueError. That error was silently caught below, which
        # is why the route kept returning "Could not parse lecture plan"
        # even for perfectly text-based PDFs.
        filled_prompt = prompt_template.replace("{text}", text[:4000])
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": filled_prompt}],
            temperature=0,
            max_tokens=1000,
        )
        raw = response.choices[0].message.content or "{}"
        raw = raw.strip()
        # Strip markdown code fences if Groq wraps in ```json ... ```
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())
        return result if isinstance(result, dict) else {}
    except Exception as e:
        # Log to stderr so uvicorn shows the real reason on future failures,
        # instead of silently returning {} and confusing the frontend.
        import traceback
        print(f"[parse_lecture_plan] failed: {e}", flush=True)
        traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# STEP 3 — Extract chapter headings from a book (PDF stored on Cloudinary)
# ---------------------------------------------------------------------------

def extract_chapter_headings_from_bytes(pdf_bytes: bytes) -> list[dict]:
    """
    Extract chapter headings WITH page numbers from a book PDF.

    Returns a list of dicts: [{"title": "Dynamic Programming", "page": 89}, ...]

    Strategy (priority order):
      1. PRIMARY: doc.get_toc() — reads the PDF's embedded Table of Contents.
         This is exact — returns level, title, and page number directly.
         Works for any digitally-created PDF (most textbooks).

      2. FALLBACK: heuristic text-block scan of first 30 pages.
         Used when get_toc() returns empty (e.g. scanned PDFs, no TOC embedded).
         Returns {"title": heading, "page": null} since page number unavailable.

    Why get_toc() first?
      The old heuristic guessed headings from text formatting — unreliable and
      returned no page numbers. get_toc() reads the actual TOC structure the
      PDF author built in, so it is both more accurate and gives exact pages.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # PRIMARY: use embedded TOC
        toc = doc.get_toc()   # returns [[level, title, page], ...]
        if toc:
            headings = []
            seen = set()
            for level, title, page in toc:
                title = title.strip()
                if title and title not in seen:
                    headings.append({"title": title, "page": page})
                    seen.add(title)
                if len(headings) >= 50:
                    break
            doc.close()
            return headings

        # FALLBACK: heuristic scan when no embedded TOC
        headings = []
        seen = set()
        for page_num, page in enumerate(doc):
            if page_num > 30:
                break
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip() if len(block) > 4 else ""
                if not text or len(text) > 100:
                    continue
                is_chapter = (
                    text.lower().startswith(("chapter", "unit", "section", "part", "module"))
                    or text[:2].isdigit()
                    or (text[0].isupper() and len(text.split()) <= 8)
                )
                if is_chapter and text not in seen:
                    headings.append({"title": text, "page": None})
                    seen.add(text)
                if len(headings) >= 30:
                    break

        doc.close()
        return headings
    except Exception:
        return []


# ---------------------------------------------------------------------------
# STEP 4 — Find best chapter match for a topic inside a specific book
# ---------------------------------------------------------------------------

def find_best_chapter(topic_vec: np.ndarray, book: dict) -> dict:
    """
    Given a topic embedding and a book, find the best matching chapter.

    chapter_headings is now a list of dicts: [{"title": str, "page": int|None}]
    Returns: {"title": str, "page": int|None, "score": float}
    """
    headings = book.get("chapter_headings", [])
    if not headings:
        return {"title": "", "page": None, "score": 0.0}

    best = {"title": "", "page": None, "score": 0.0}
    for h in headings:
        # support both old format (plain string) and new format (dict)
        if isinstance(h, dict):
            h_title = h.get("title", "")
            h_page  = h.get("page", None)
        else:
            h_title = str(h)
            h_page  = None

        if not h_title:
            continue

        h_vec   = np.array([generate_embedding(h_title)])
        h_score = float(cosine_similarity(topic_vec, h_vec)[0][0])

        if h_score > best["score"]:
            best = {"title": h_title, "page": h_page, "score": round(h_score, 4)}

    return best


# ---------------------------------------------------------------------------
# STEP 5 — Find professor-recommended books in inventory (vector search)
# ---------------------------------------------------------------------------

PROFESSOR_BOOK_MATCH_THRESHOLD = 0.60   # cosine sim threshold for "found in library"

def _normalise_recommended_entry(entry) -> dict:
    """
    Groq may return a structured dict {title, authors} (new format) OR a plain
    string (old format, or if the LLM ignored instructions). Normalise both to
    {title, authors} so the rest of the pipeline has one shape to deal with.

    String fallback heuristic: the longest comma-separated chunk that doesn't
    look like a publisher/year is probably the title; the rest we treat as
    authors. This is only a safety net — the prompt should usually return dicts.
    """
    if isinstance(entry, dict):
        return {
            "title":   (entry.get("title") or "").strip(),
            "authors": (entry.get("authors") or "").strip(),
        }

    # Legacy string handling — best-effort split
    s = str(entry or "").strip()
    if not s:
        return {"title": "", "authors": ""}

    # common publishers/editions we want to strip
    noise = ("wiley", "pearson", "springer", "mcgraw", "phi", "oxford",
             "sultan chand", "prentice", "cengage", "tmh", "edition", "ed.")
    parts = [p.strip() for p in s.replace(":", ",").split(",") if p.strip()]
    parts = [p for p in parts
             if not p.isdigit()
             and not any(n in p.lower() for n in noise)]

    if not parts:
        return {"title": s, "authors": ""}
    # assume longest chunk is the title
    title = max(parts, key=len)
    authors = ", ".join(p for p in parts if p != title)
    return {"title": title, "authors": authors}


def find_recommended_books_in_inventory(
    recommended: list,
    books: list[dict],
    pdfs: list[dict] | None = None,
) -> list[dict]:
    """
    For each professor-recommended book (from lecture plan page 2), search BOTH
    the physical-book inventory AND the digital-PDF inventory using vector
    similarity on title+author text. Best match across both wins.

    Why search both:
      The professor writes "Operating Systems Principles — Silberschatz, Galvin".
      A student may have uploaded the PDF as "Operating System Concepts — Galvin".
      Those are the same textbook. If we only searched `books`, the student's
      uploaded PDF would never count as "professor's book found" — even though
      it has the exact chapter structure we need for the Unit → Chapter map.

    `recommended` accepts two shapes:
      - New (preferred): [{"title": "...", "authors": "..."}, ...]
      - Legacy: ["Author: Title, Publisher, Year", ...]  — auto-normalised.

    Returns list of:
      {
        "recommended_title":   str,   # clean title extracted from plan (for display)
        "recommended_authors": str,
        "found":           bool,
        "source_type":     "physical" | "pdf" | "",    # which collection matched
        "book_title":      str,       # from matched document
        "book_author":     str,
        "book_id":         str,       # physical book ObjectId OR pdf ObjectId
        "cloudinary_url":  str,       # only set when source_type == "pdf"
        "available":       bool,      # physical: live; pdf: always True (downloadable)
        "match_score":     float
      }

    Note: `book_id` holds the ID regardless of source_type — downstream
    map_units_to_chapters already splits the flat `preferred_book_ids` list
    into books vs pdfs by checking each collection, so a PDF id flows through
    naturally.
    """
    books_with_vectors = [b for b in books if b.get("vector")]
    pdfs_with_vectors  = [p for p in (pdfs or []) if p.get("embedding")]
    results = []

    for entry in recommended:
        rec = _normalise_recommended_entry(entry)
        rec_title   = rec["title"]
        rec_authors = rec["authors"]
        if not rec_title:
            continue

        # Embed title + authors together — cleaner signal than raw citation
        # (which had publishers, years, edition numbers as noise).
        embed_text = f"{rec_title} {rec_authors}".strip()
        rec_vec = np.array([generate_embedding(embed_text)])

        best_score  = 0.0
        best_item   = None
        best_type   = ""   # "physical" | "pdf"

        # --- Search physical books ---
        for book in books_with_vectors:
            book_vec = np.array([book["vector"]])
            score    = float(cosine_similarity(rec_vec, book_vec)[0][0])
            if score > best_score:
                best_score = score
                best_item  = book
                best_type  = "physical"

        # --- Search digital PDFs (student-uploaded soft copies) ---
        for pdf in pdfs_with_vectors:
            pdf_vec = np.array([pdf["embedding"]])
            score   = float(cosine_similarity(rec_vec, pdf_vec)[0][0])
            if score > best_score:
                best_score = score
                best_item  = pdf
                best_type  = "pdf"

        if best_item and best_score >= PROFESSOR_BOOK_MATCH_THRESHOLD:
            # PDFs don't have an `author` field — fall back to `professor` (uploader)
            # or leave blank. The professor-recommended authors still show via
            # `recommended_authors` in the card subtitle.
            item_author = (
                best_item.get("author", "") if best_type == "physical"
                else best_item.get("professor", "")
            )
            results.append({
                "recommended_title":   rec_title,
                "recommended_authors": rec_authors,
                "found":           True,
                "source_type":     best_type,
                "book_title":      best_item.get("title", ""),
                "book_author":     item_author,
                "book_id":         str(best_item.get("_id", "")),
                "cloudinary_url":  best_item.get("cloudinary_url", "") if best_type == "pdf" else "",
                "available":       (
                    best_item.get("available", True) if best_type == "physical"
                    else True   # PDFs are always downloadable
                ),
                "match_score":     round(best_score, 4)
            })
        else:
            results.append({
                "recommended_title":   rec_title,
                "recommended_authors": rec_authors,
                "found":           False,
                "source_type":     "",
                "book_title":      "",
                "book_author":     "",
                "book_id":         "",
                "cloudinary_url":  "",
                "available":       False,
                "match_score":     round(best_score, 4) if best_item else 0.0
            })

    return results


# ---------------------------------------------------------------------------
# STEP 5b — Tier 3 chapter fallback: Groq guesses probable chapter from metadata
# ---------------------------------------------------------------------------
#
# Thesis framing (Chapter 3): 3-tier chapter source ladder
#   Tier 1: PDF embedded TOC via doc.get_toc() — exact page numbers
#   Tier 2: Physical book heuristic text scan — chapter titles only, no pages
#   Tier 3: Groq LLM estimate from book title + author + unit topic — labeled as guess
#
# Fallback triggers when Tier 1 and Tier 2 return empty OR low-confidence match.
# All Tier 3 output is labeled "ai_guess" so frontend never claims accuracy it
# doesn't have. This is the honesty principle that defines thesis-grade systems.

CHAPTER_CONFIDENCE_THRESHOLD = 0.50   # below this, trigger Tier 3 estimate

CHAPTER_ESTIMATE_PROMPT = """For the textbook "{title}" by {author}, which chapter most likely covers the topic "{unit_title}"?

Base your answer on standard chapter structure for textbooks on this subject.

Return ONLY valid JSON (no markdown, no explanation):
{{
  "chapter_title": "e.g. 'Chapter 6: Dynamic Programming'",
  "confidence": "low" | "medium"
}}

If you don't recognize the book or can't reasonably guess, return:
{{"chapter_title": "", "confidence": "low"}}"""


def estimate_chapter_with_groq(book_title: str, book_author: str, unit_title: str) -> dict:
    """
    Tier 3 fallback — ask Groq to guess a probable chapter name from its training data.

    Used ONLY when Tiers 1 and 2 fail:
      - Book has no chapter_headings (no PDF, no heuristic hits)
      - OR find_best_chapter returned score < 0.5 (weak match)

    Returns: {"title": str, "page": None, "confidence": "low"|"medium"}
             page is ALWAYS None — LLM cannot know real pagination.

    Why this is thesis-defensible:
      Standard textbooks follow conventional chapter orderings (Operations Research
      books always cover LP → Simplex → Duality → Transportation → Assignment →
      Network → DP → Queuing, roughly). Groq has seen thousands of these — its
      guess is informed, not random. But we NEVER claim accuracy: output is
      explicitly labeled as an estimate.
    """
    if not book_title or not unit_title:
        return {"title": "", "page": None, "confidence": "low"}

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": CHAPTER_ESTIMATE_PROMPT.format(
                    title=book_title,
                    author=book_author or "Unknown",
                    unit_title=unit_title,
                )
            }],
            temperature=0,        # reproducibility — same book+topic ⇒ same guess
            max_tokens=150,
        )
        raw = (response.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        return {
            "title":      (data.get("chapter_title") or "").strip(),
            "page":       None,
            "confidence": data.get("confidence", "low"),
        }
    except Exception:
        return {"title": "", "page": None, "confidence": "low"}


# ---------------------------------------------------------------------------
# STEP 6 — Map each unit to the best book + chapter (with twin detection)
# ---------------------------------------------------------------------------

TWIN_MATCH_THRESHOLD = 0.75   # cosine sim at which physical ≡ PDF counts as same book


def _find_twin(
    chosen_book: dict,
    chosen_type: str,
    books_with_vectors: list[dict],
    pdfs_with_vectors: list[dict]
) -> dict | None:
    """
    Given a chosen listing (physical book or PDF), find the same book in the OTHER
    collection. Returns the twin dict or None.

    Why: if the professor recommends "Operations Research by Kanti Swarup" and we
    have BOTH a physical copy AND a PDF upload, the student should see both formats
    — borrow the hard copy OR download the PDF. Picking one silently loses info.
    """
    if chosen_type == "physical":
        chosen_vec = np.array([chosen_book["vector"]])
        target_pool = pdfs_with_vectors
        vec_key = "embedding"
    else:
        chosen_vec = np.array([chosen_book["embedding"]])
        target_pool = books_with_vectors
        vec_key = "vector"

    best_score = 0.0
    best_twin  = None
    for other in target_pool:
        if str(other.get("_id", "")) == str(chosen_book.get("_id", "")):
            continue
        other_vec = np.array([other[vec_key]])
        score = float(cosine_similarity(chosen_vec, other_vec)[0][0])
        if score > best_score:
            best_score = score
            best_twin  = other

    return best_twin if (best_twin and best_score >= TWIN_MATCH_THRESHOLD) else None


def map_units_to_chapters(
    units: list[dict],
    books: list[dict],
    preferred_book_ids: list[str],  # professor-recommended books found in inventory
    pdfs: list[dict] = None         # digital PDFs — have accurate TOC page numbers
) -> list[dict]:
    """
    For each unit in the lecture plan, find the best book + chapter.

    Priority ladder:
      1. Professor-recommended books (physical OR PDF — whichever matches unit better)
      2. Fallback: any book/PDF in inventory

    Twin detection: once a primary book is chosen, search the OTHER collection for
    a counterpart. If found (cosine ≥ 0.75), both formats are returned in
    `formats_available` — student picks format.

    Chapter extraction: if PDF exists for the book, chapter headings come from it
    (accurate page numbers via get_toc). Otherwise physical book's headings.

    Returns list of:
      {
        "unit_no": int,
        "unit_title": str,
        "book_title": str,
        "book_author": str,
        "source": "professor_recommended" | "ai_suggested",
        "match_score": float,
        "formats_available": [
          {"type": "physical", "book_id": str, "available": bool},
          {"type": "pdf", "pdf_id": str, "cloudinary_url": str, "page_accurate": True}
        ],
        "suggested_chapter": str,
        "chapter_page": int | None,
        "chapter_source": "pdf" | "physical"   # "pdf" = page number reliable
      }
    """
    books_with_vectors = [b for b in books if b.get("vector")]
    pdfs_with_vectors  = [p for p in (pdfs or []) if p.get("embedding")]

    preferred_books = [b for b in books_with_vectors
                       if str(b.get("_id", "")) in preferred_book_ids]
    preferred_pdfs  = [p for p in pdfs_with_vectors
                       if str(p.get("_id", "")) in preferred_book_ids]

    results = []

    for unit in units:
        unit_title = unit.get("title", "")
        if not unit_title:
            continue

        topic_vec = np.array([generate_embedding(unit_title)])

        chosen_book  = None
        chosen_score = 0.0
        chosen_type  = "physical"
        source       = "ai_suggested"

        # --- Priority 1: professor-recommended (physical OR PDF) ---
        for book in preferred_books:
            book_vec = np.array([book["vector"]])
            score = float(cosine_similarity(topic_vec, book_vec)[0][0])
            if score > chosen_score:
                chosen_score = score
                chosen_book  = book
                chosen_type  = "physical"
                source       = "professor_recommended"

        for pdf in preferred_pdfs:
            pdf_vec = np.array([pdf["embedding"]])
            score = float(cosine_similarity(topic_vec, pdf_vec)[0][0])
            if score > chosen_score:
                chosen_score = score
                chosen_book  = pdf
                chosen_type  = "pdf"
                source       = "professor_recommended"

        # --- Priority 2: fallback — any physical or PDF in inventory ---
        if not chosen_book:
            for book in books_with_vectors:
                book_vec = np.array([book["vector"]])
                score = float(cosine_similarity(topic_vec, book_vec)[0][0])
                if score > chosen_score:
                    chosen_score = score
                    chosen_book  = book
                    chosen_type  = "physical"
                    source       = "ai_suggested"

            for pdf in pdfs_with_vectors:
                pdf_vec = np.array([pdf["embedding"]])
                score = float(cosine_similarity(topic_vec, pdf_vec)[0][0])
                if score > chosen_score:
                    chosen_score = score
                    chosen_book  = pdf
                    chosen_type  = "pdf"
                    source       = "ai_suggested"

        if not chosen_book:
            continue

        # --- Twin detection: same book in the OTHER collection? ---
        twin = _find_twin(chosen_book, chosen_type,
                          books_with_vectors, pdfs_with_vectors)

        # --- Build formats_available list (what student can access) ---
        formats = []
        if chosen_type == "physical":
            formats.append({
                "type":      "physical",
                "book_id":   str(chosen_book.get("_id", "")),
                "available": chosen_book.get("available", True),
            })
            if twin:
                formats.append({
                    "type":           "pdf",
                    "pdf_id":         str(twin.get("_id", "")),
                    "cloudinary_url": twin.get("cloudinary_url", ""),
                    "page_accurate":  True,
                })
        else:  # chosen_type == "pdf"
            formats.append({
                "type":           "pdf",
                "pdf_id":         str(chosen_book.get("_id", "")),
                "cloudinary_url": chosen_book.get("cloudinary_url", ""),
                "page_accurate":  True,
            })
            if twin:
                formats.append({
                    "type":      "physical",
                    "book_id":   str(twin.get("_id", "")),
                    "available": twin.get("available", True),
                })

        # --- Pick chapter source: prefer PDF (get_toc → real page numbers) ---
        chapter_source_book = chosen_book
        chapter_source_type = chosen_type
        if chosen_type == "physical" and twin and twin.get("chapter_headings"):
            chapter_source_book = twin
            chapter_source_type = "pdf"

        chapter = find_best_chapter(topic_vec, chapter_source_book)

        # --- Tier 3: Groq fallback if TOC/heuristic gave weak or no match ---
        chapter_confidence = "high"  # "high" = TOC/heuristic hit, "low"/"medium" = AI guess
        if not chapter["title"] or chapter["score"] < CHAPTER_CONFIDENCE_THRESHOLD:
            estimate = estimate_chapter_with_groq(
                chosen_book.get("title", ""),
                chosen_book.get("author", "") or chosen_book.get("professor", ""),
                unit_title,
            )
            if estimate["title"]:
                chapter = {"title": estimate["title"], "page": None, "score": 0.0}
                chapter_source_type = "ai_guess"
                chapter_confidence  = estimate["confidence"]
            # If Groq also returned nothing, leave chapter empty — frontend will say
            # "no chapter info available, book is still listed as available."

        results.append({
            "unit_no":            unit.get("unit_no", ""),
            "unit_title":         unit_title,
            "book_title":         chosen_book.get("title", ""),
            "book_author":        chosen_book.get("author", "") or chosen_book.get("professor", ""),
            "source":             source,
            "match_score":        round(chosen_score, 4),
            "formats_available":  formats,
            "suggested_chapter":  chapter["title"],
            "chapter_page":       chapter["page"],
            "chapter_source":     chapter_source_type,   # "pdf" | "physical" | "ai_guess"
            "chapter_confidence": chapter_confidence,    # "high" | "medium" | "low"
        })

    return results


# ---------------------------------------------------------------------------
# STEP 7 — Full pipeline entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# STEP 6 — Course-tagged notes search (complements unit matching)
# ---------------------------------------------------------------------------
#
# Motivation: when the professor's recommended textbooks aren't in our
# inventory AND no PDF matches a unit by semantic similarity, the result
# page would be empty and useless. BUT — students often upload notes,
# previous year papers, and reference material tagged to the same course
# or department. Those still help.
#
# This search is metadata-only (no embeddings needed): filter `pdfs`
# where course / subject / department match the lecture plan's course_name,
# course_code, or department. Group results by resource_type so the UI can
# render Notes / Previous Papers / Reference as separate rows.
#
# Thesis framing (Chapter 3): complements semantic unit→chapter matching
# with a "peer-uploaded material" retrieval path — same query, different
# retrieval signal (structured metadata vs dense vector similarity).

def find_course_related_notes(
    course_name: str,
    course_code: str,
    department: str,
    pdfs: list[dict] | None,
    limit_per_type: int = 3,
) -> list[dict]:
    """
    Find PDFs tagged to the lecture plan's course or department.

    Match on any of (case-insensitive substring):
      - pdf.course        ⊇ course_code  or  course_name
      - pdf.subject       ⊇ course_name
      - pdf.department    ⊇ department

    Returns flat list (grouped/capped per resource_type) — UI groups for display.
    """
    if not pdfs:
        return []

    course_name_lc = (course_name or "").strip().lower()
    course_code_lc = (course_code or "").strip().lower()
    department_lc  = (department or "").strip().lower()

    # Nothing to match against → nothing to return
    if not (course_name_lc or course_code_lc or department_lc):
        return []

    matches = []
    for pdf in pdfs:
        pdf_course  = (pdf.get("course", "") or "").lower()
        pdf_subject = (pdf.get("subject", "") or "").lower()
        pdf_dept    = (pdf.get("department", "") or "").lower()

        # Try strongest signals first so match_reason reflects the best hit.
        match_reason = None
        if course_code_lc and pdf_course and course_code_lc in pdf_course:
            match_reason = "course_code"
        elif course_name_lc and (
            (pdf_course and course_name_lc in pdf_course) or
            (pdf_subject and course_name_lc in pdf_subject)
        ):
            match_reason = "course_name"
        elif department_lc and pdf_dept and department_lc in pdf_dept:
            match_reason = "department"

        if not match_reason:
            continue

        matches.append({
            "pdf_id":         str(pdf.get("_id", "")),
            "title":          pdf.get("title", ""),
            "subject":        pdf.get("subject", ""),
            "course":         pdf.get("course", ""),
            "department":     pdf.get("department", ""),
            "professor":      pdf.get("professor", ""),
            "resource_type":  pdf.get("resource_type", "notes"),
            "cloudinary_url": pdf.get("cloudinary_url", ""),
            "match_reason":   match_reason,
        })

    # Cap per resource_type so one flooded category doesn't crowd others.
    by_type: dict[str, list[dict]] = {}
    for m in matches:
        by_type.setdefault(m["resource_type"], []).append(m)

    trimmed = []
    for items in by_type.values():
        trimmed.extend(items[:limit_per_type])

    return trimmed


def process_curriculum_pdf(
    pdf_bytes: bytes,
    books: list[dict],
    pdfs: list[dict] = None,
    *,
    prompt_version: str | None = None,
    pdf_filename: str | None = None,
) -> dict:
    """
    Full pipeline: lecture plan PDF bytes → structured curriculum match.

    Args:
      pdf_bytes       raw PDF bytes (as received from the upload form)
      books           current physical-book inventory (list of Book docs)
      pdfs            current digital-PDF inventory (list of Pdf docs)
      prompt_version  which curriculum prompt to use ("v1", "v2"); defaults
                      to latest from app/prompts/__init__.py
      pdf_filename    original upload filename — logged for reproducibility

    Every call (success or failure) appends one row to
    experiments/curriculum_runs.jsonl so Chapter 5 tables can be built by
    reading that log without re-running the pipeline.

    Returns:
      {
        "course_name", "course_code", "department",
        "recommended_books": [...],
        "unit_matches":      [...],
        "prompt_version":    str,   ← echoed back so the caller can log
        "error":             str | None
      }
    """
    active_version = prompt_version or CURRICULUM_LATEST

    # Extract raw text from PDF
    raw_text = extract_text_from_bytes(pdf_bytes)
    if raw_text.startswith("ERROR:"):
        # Still log the failed attempt — useful to know which PDFs are scanned/broken.
        log_curriculum_run(
            prompt_version=active_version,
            pdf_filename=pdf_filename,
            raw_text_len=0,
            parsed={},
            recommended_books=[],
            unit_matches=[],
            error=raw_text,
        )
        return {"error": raw_text, "prompt_version": active_version}

    # One Groq call to parse everything: units + textbooks + reference books
    parsed = parse_lecture_plan(raw_text, prompt_version=active_version)
    if not parsed:
        log_curriculum_run(
            prompt_version=active_version,
            pdf_filename=pdf_filename,
            raw_text_len=len(raw_text),
            parsed={},
            recommended_books=[],
            unit_matches=[],
            error="Could not parse lecture plan (LLM returned empty/invalid JSON)",
        )
        return {
            "error": "Could not parse lecture plan. Is it a text-based PDF?",
            "prompt_version": active_version,
        }

    units     = parsed.get("units", [])
    textbooks = parsed.get("textbooks", []) + parsed.get("reference_books", [])

    # Step A: find professor's recommended books in our inventory.
    # Search BOTH physical and PDF collections — a student-uploaded PDF of
    # Silberschatz counts as "the professor's book is in our library".
    recommended_books = find_recommended_books_in_inventory(textbooks, books, pdfs)

    # IDs of recommended books that were actually found in library
    found_ids = [r["book_id"] for r in recommended_books if r["found"]]

    # Step B: map each unit to best chapter (professor books first, fallback second)
    unit_matches = map_units_to_chapters(units, books, preferred_book_ids=found_ids, pdfs=pdfs)

    # Step C: peer-uploaded notes / PYQs / reference tagged to this course or dept.
    # Runs regardless of unit-matching outcome — surfaces material even when the
    # semantic matcher finds nothing (common on the day a new lecture plan is
    # uploaded and no student has posted a matching textbook yet).
    related_notes = find_course_related_notes(
        course_name=parsed.get("course_name", ""),
        course_code=parsed.get("course_code", ""),
        department=parsed.get("department", ""),
        pdfs=pdfs,
    )

    # Append run record for thesis Chapter 5 — happens AFTER all matching so we
    # capture the full end-to-end outcome, not just what the LLM parsed.
    log_curriculum_run(
        prompt_version=active_version,
        pdf_filename=pdf_filename,
        raw_text_len=len(raw_text),
        parsed=parsed,
        recommended_books=recommended_books,
        unit_matches=unit_matches,
        error=None,
    )

    return {
        "course_name":       parsed.get("course_name", ""),
        "course_code":       parsed.get("course_code", ""),
        "department":        parsed.get("department", ""),
        "recommended_books": recommended_books,
        "unit_matches":      unit_matches,
        "related_notes":     related_notes,   # PDFs tagged to this course/dept
        # n_units_parsed lets the UI tell apart two failure modes:
        #   (a) Groq couldn't parse any units from the PDF (scanned, garbled)
        #   (b) Units parsed fine but inventory had no matching book/PDF
        # Without this, both show the same "couldn't extract units" message
        # and we misdiagnose inventory gaps as parse failures.
        "n_units_parsed":    len(parsed.get("units", []) or []),
        "prompt_version":    active_version,
        "error":             None,
    }

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

LECTURE_PLAN_PARSE_PROMPT = """You are parsing an IIT ISM Dhanbad standard lecture plan PDF.
These PDFs always follow the same 2-page format:
  Page 1: Course details + a table of units with topics
  Page 2: Textbooks and Reference Books listed by the professor

Extract ALL of the following and return ONLY valid JSON. No explanation, no markdown.

Return this exact structure:
{
  "course_name": "string — name of the course",
  "course_code": "string — e.g. MCO502",
  "department": "string — department name",
  "units": [
    {
      "unit_no": 1,
      "title": "short unit title e.g. Network Analysis",
      "topics": ["topic 1", "topic 2"]
    }
  ],
  "textbooks": ["Author: Title, Publisher, Year", ...],
  "reference_books": ["Author: Title, Publisher, Year", ...]
}

Lecture plan text:
{text}"""


def parse_lecture_plan(text: str) -> dict:
    """
    Parse a complete IIT ISM lecture plan PDF text into structured data.

    Returns a dict with:
      course_name, course_code, department,
      units: [{unit_no, title, topics[]}],
      textbooks: [str],
      reference_books: [str]

    Why one Groq call instead of two?
      The old approach made one call for topics only — ignoring the book list
      on page 2 entirely. The professor's recommended books are ground truth.
      We should check those first before doing any semantic search.
      One structured call extracts everything we need in one shot.
    """
    if not text or len(text.strip()) < 50:
        return {}

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": LECTURE_PLAN_PARSE_PROMPT.format(text=text[:4000])
            }],
            temperature=0,
            max_tokens=1000
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
    except Exception:
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

def find_recommended_books_in_inventory(
    recommended: list[str],
    books: list[dict]
) -> list[dict]:
    """
    For each professor-recommended book string (from lecture plan page 2),
    search the inventory using vector similarity on title+author text.

    Why vectors and not exact string match?
      The professor writes: "Kanti Swarup, P.K. Gupta: Operations Research, Sultan Chand, 2017"
      The student posted it as: "Operations Research"
      Exact match fails. Vector similarity finds it.

    Returns list of:
      {
        "recommended_title": str,   # what the professor wrote
        "found": bool,
        "book_title": str,          # what's in our DB (if found)
        "book_author": str,
        "book_id": str,
        "available": bool,
        "match_score": float
      }
    """
    books_with_vectors = [b for b in books if b.get("vector")]
    results = []

    for rec_title in recommended:
        rec_vec = np.array([generate_embedding(rec_title)])

        best_score = 0.0
        best_book  = None
        for book in books_with_vectors:
            book_vec = np.array([book["vector"]])
            score    = float(cosine_similarity(rec_vec, book_vec)[0][0])
            if score > best_score:
                best_score = score
                best_book  = book

        if best_book and best_score >= PROFESSOR_BOOK_MATCH_THRESHOLD:
            results.append({
                "recommended_title": rec_title,
                "found":       True,
                "book_title":  best_book.get("title", ""),
                "book_author": best_book.get("author", ""),
                "book_id":     str(best_book.get("_id", "")),
                "available":   best_book.get("available", True),
                "match_score": round(best_score, 4)
            })
        else:
            results.append({
                "recommended_title": rec_title,
                "found":       False,
                "book_title":  "",
                "book_author": "",
                "book_id":     "",
                "available":   False,
                "match_score": round(best_score, 4) if best_book else 0.0
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

def process_curriculum_pdf(pdf_bytes: bytes, books: list[dict], pdfs: list[dict] = None) -> dict:
    """
    Full pipeline: lecture plan PDF bytes → structured curriculum match.

    Returns:
      {
        "course_name": str,
        "course_code": str,
        "department": str,
        "recommended_books": [   ← professor's books, found/not found in library
          {"recommended_title", "found", "book_title", "available", "match_score", ...}
        ],
        "unit_matches": [        ← each unit → best book + chapter + page
          {"unit_no", "unit_title", "book_title", "source", "suggested_chapter",
           "chapter_page", "available", "match_score", ...}
        ],
        "error": str | None
      }
    """
    # Extract raw text from PDF
    raw_text = extract_text_from_bytes(pdf_bytes)
    if raw_text.startswith("ERROR:"):
        return {"error": raw_text}

    # One Groq call to parse everything: units + textbooks + reference books
    parsed = parse_lecture_plan(raw_text)
    if not parsed:
        return {"error": "Could not parse lecture plan. Is it a text-based PDF?"}

    units    = parsed.get("units", [])
    textbooks = parsed.get("textbooks", []) + parsed.get("reference_books", [])

    # Step A: find professor's recommended books in our inventory
    recommended_books = find_recommended_books_in_inventory(textbooks, books)

    # IDs of recommended books that were actually found in library
    found_ids = [r["book_id"] for r in recommended_books if r["found"]]

    # Step B: map each unit to best chapter (professor books first, fallback second)
    unit_matches = map_units_to_chapters(units, books, preferred_book_ids=found_ids, pdfs=pdfs)

    return {
        "course_name":       parsed.get("course_name", ""),
        "course_code":       parsed.get("course_code", ""),
        "department":        parsed.get("department", ""),
        "recommended_books": recommended_books,
        "unit_matches":      unit_matches,
        "error":             None
    }

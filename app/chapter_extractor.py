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

TOPIC_PARSE_PROMPT = """You are parsing an IIT ISM Dhanbad lecture plan PDF.
Extract ALL lecture topics/units from the text below.
Return a JSON array of strings — each string is one topic/unit title.
Be concise. No explanations. No extra text. Only valid JSON array.

Example output: ["Introduction to Operating Systems", "Process Scheduling", "Memory Management"]

Lecture plan text:
{text}"""

def parse_topics_from_text(text: str) -> list[str]:
    """
    Use Groq (LLM) to parse structured topics from raw extracted PDF text.

    Returns:
      List of topic strings, e.g. ["Process Scheduling", "Deadlock Detection", ...]
      Empty list if parsing fails or text is too short.
    """
    if not text or len(text.strip()) < 50:
        return []

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": TOPIC_PARSE_PROMPT.format(text=text[:3000])
            }],
            temperature=0,
            max_tokens=500
        )
        raw = response.choices[0].message.content or "[]"
        # Strip markdown code blocks if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        topics = json.loads(raw.strip())
        if isinstance(topics, list):
            return [str(t).strip() for t in topics if t]
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# STEP 3 — Extract chapter headings from a book (PDF stored on Cloudinary)
# ---------------------------------------------------------------------------

def extract_chapter_headings_from_bytes(pdf_bytes: bytes) -> list[str]:
    """
    Extract potential chapter headings from a book PDF.

    Strategy:
      1. Use PyMuPDF to extract text block by block
      2. Filter blocks that look like chapter headings:
         - Short (< 80 chars)
         - Start with "Chapter", "Unit", "Section", a number, or uppercase
      3. Return deduplicated list of up to 30 headings

    Used in Phase 10 to map lecture topics to specific book chapters.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        headings = []
        seen = set()

        for page_num, page in enumerate(doc):
            if page_num > 30:   # only scan first 30 pages (TOC area)
                break
            blocks = page.get_text("blocks")
            for block in blocks:
                text = block[4].strip() if len(block) > 4 else ""
                if not text or len(text) > 100:
                    continue
                # Heading heuristics
                is_chapter = (
                    text.lower().startswith(("chapter", "unit", "section", "part", "module"))
                    or text[:2].isdigit()
                    or (text[0].isupper() and len(text.split()) <= 8)
                )
                if is_chapter and text not in seen:
                    headings.append(text)
                    seen.add(text)
                if len(headings) >= 30:
                    break

        doc.close()
        return headings
    except Exception:
        return []


# ---------------------------------------------------------------------------
# STEP 4 — Match topics to books + find best chapter
# ---------------------------------------------------------------------------

def match_topics_to_books(topics: list[str], books: list[dict]) -> list[dict]:
    """
    For each topic, find the most relevant book and best matching chapter heading.

    Args:
      topics: list of topic strings from the lecture plan
      books:  list of book dicts (must have 'vector', 'title', 'author', '_id')

    Returns:
      List of match dicts:
        {
          "topic": str,
          "book_title": str,
          "book_author": str,
          "book_id": str,
          "match_score": float,     # cosine sim between topic and book embedding
          "suggested_chapter": str  # best chapter heading from book PDF (if available)
          "chapter_score": float
        }

    Chapter matching uses title+description embedding if no chapter headings available.
    """
    if not topics or not books:
        return []

    books_with_vectors = [b for b in books if b.get("vector")]
    if not books_with_vectors:
        return []

    results = []

    for topic in topics:
        topic_vec = np.array([generate_embedding(topic)])

        # Score all books against this topic
        scored = []
        for book in books_with_vectors:
            book_vec = np.array([book["vector"]])
            score = float(cosine_similarity(topic_vec, book_vec)[0][0])
            scored.append((score, book))

        scored.sort(reverse=True)
        top_score, top_book = scored[0]

        # Try to suggest a chapter if book has chapter_headings stored
        suggested_chapter = ""
        chapter_score = 0.0
        headings = top_book.get("chapter_headings", [])
        if headings:
            best_ch, best_score = "", 0.0
            for heading in headings:
                h_vec = np.array([generate_embedding(heading)])
                h_score = float(cosine_similarity(topic_vec, h_vec)[0][0])
                if h_score > best_score:
                    best_score = h_score
                    best_ch = heading
            suggested_chapter = best_ch
            chapter_score = round(best_score, 4)

        results.append({
            "topic": topic,
            "book_title": top_book.get("title", "Unknown"),
            "book_author": top_book.get("author", ""),
            "book_id": str(top_book.get("_id", "")),
            "match_score": round(top_score, 4),
            "suggested_chapter": suggested_chapter,
            "chapter_score": chapter_score
        })

    return results


# ---------------------------------------------------------------------------
# STEP 5 — Full pipeline: PDF bytes → topic matches
# (single entry point used by the /curriculum endpoint)
# ---------------------------------------------------------------------------

def process_curriculum_pdf(pdf_bytes: bytes, books: list[dict]) -> dict:
    """
    Full pipeline: PDF bytes → structured topic-to-book matches.

    Returns:
      {
        "topics_found": int,
        "topics": [str],
        "matches": [match_dict],
        "raw_text_preview": str   # first 200 chars for debugging
      }
    """
    raw_text = extract_text_from_bytes(pdf_bytes)

    if raw_text.startswith("ERROR:"):
        return {"error": raw_text, "topics_found": 0, "topics": [], "matches": []}

    topics = parse_topics_from_text(raw_text)

    if not topics:
        return {
            "error": "Could not extract topics from this PDF. Is it a scanned image?",
            "topics_found": 0,
            "topics": [],
            "matches": [],
            "raw_text_preview": raw_text[:200]
        }

    matches = match_topics_to_books(topics, books)

    return {
        "topics_found": len(topics),
        "topics": topics,
        "matches": matches,
        "raw_text_preview": raw_text[:200]
    }

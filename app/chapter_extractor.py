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


# ---------------------------------------------------------------------------
# Tesseract / tessdata auto-discovery for OCR fallback
# ---------------------------------------------------------------------------
# PyMuPDF's get_textpage_ocr() needs the path to Tesseract's tessdata
# directory. It checks TESSDATA_PREFIX env var first, then gives up. On
# Windows the env var is often not propagated into the shell that runs
# uvicorn (set in a different session, set system-wide but not picked up,
# etc.) which manifests as the misleading error message:
#
#   RuntimeError: No tessdata specified and Tesseract is not installed
#
# We resolve the path ourselves at startup by checking the env var first
# and then falling back to the standard install locations on each OS.
# The resolved path (or None) is passed explicitly to get_textpage_ocr so
# the call works regardless of whether the user remembered to set the env
# variable. If nothing is found, OCR raises and we fall through to the
# scanned-image error message exactly as before.
def _find_tessdata_path() -> str | None:
    """Return the first tessdata directory that exists on this system."""
    env = os.environ.get("TESSDATA_PREFIX")
    if env and os.path.isdir(env):
        return env
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        "/usr/share/tesseract-ocr/5/tessdata",
        "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tesseract-ocr/tessdata",
        "/opt/homebrew/share/tessdata",      # macOS apple silicon
        "/usr/local/share/tessdata",          # macOS intel
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


_TESSDATA_PATH = _find_tessdata_path()
if _TESSDATA_PATH:
    print(f"[OCR] tessdata located at: {_TESSDATA_PATH}", flush=True)
else:
    print("[OCR] tessdata NOT found — scanned PDFs will fall through to friendly error", flush=True)
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


def extract_text_from_bytes(pdf_bytes: bytes, *, allow_ocr: bool = True) -> str:
    """Same as extract_text_from_pdf but accepts bytes (for in-memory uploads).

    Two-tier extraction (added after user testing surfaced scanned lecture
    plans being silently rejected with a generic "could not parse" error):

      Tier 1 — PyMuPDF native text extraction. Works on all born-digital
               PDFs. ~5-30 ms per page.

      Tier 2 — OCR fallback for pages that returned ≤20 chars natively.
               Uses page.get_textpage_ocr() which dispatches to Tesseract
               under the hood. Adds 2-8 s per page depending on image
               density. Triggered per-page so a partially-scanned PDF
               (cover scanned, body native text) still extracts cleanly.

    OCR requires the Tesseract binary on the host. When Tesseract is
    missing, page.get_textpage_ocr() raises; we catch and fall through to
    the empty-string path, and the upstream caller emits the friendly
    "scanned image, can't read" error message. So OCR is purely additive:
    if it works, scanned PDFs become readable; if it doesn't, the system
    behaves exactly as before.

    Set allow_ocr=False to force native-only (used by callers that don't
    want the OCR latency, e.g. embed-pdf hot path on upload).
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        return f"ERROR: Could not read PDF — {str(e)}"

    pages = []
    try:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            # Tier 2 — OCR fallback for empty / near-empty native extractions
            if allow_ocr and len(text) < 20:
                print(f"[OCR] page {page_num}: native text len={len(text)}, attempting Tesseract OCR…", flush=True)
                try:
                    # tessdata=... is the key fix. Without it, PyMuPDF only
                    # checks the TESSDATA_PREFIX env var, which is brittle
                    # on Windows. We pass the resolved path directly.
                    #
                    # DPI tuning: Tesseract's peak memory scales roughly
                    # quadratically with DPI. 300 DPI on a 1-page A4 scan
                    # spikes to ~120 MB; 150 DPI to ~50 MB. The drop is
                    # what lets the call fit under Render's free-tier 512 MB
                    # cap (steady state with model loaded is ~380 MB, so
                    # 300 DPI tipped over while 150 DPI does not). Recall
                    # quality on lecture-plan-style scans (typed body text,
                    # no handwriting, decent contrast) is essentially
                    # identical between 150 and 300 DPI. Bumped back up
                    # to 300 if/when the service moves to a paid plan
                    # with more RAM headroom — the env var below makes
                    # that a one-line change.
                    ocr_dpi = int(os.environ.get("OCR_DPI", "150"))
                    tp = page.get_textpage_ocr(
                        flags=0,
                        language="eng",
                        dpi=ocr_dpi,
                        full=True,          # OCR the whole page
                        tessdata=_TESSDATA_PATH,
                    )
                    ocr_text = page.get_text("text", textpage=tp).strip()
                    print(f"[OCR] page {page_num}: OCR returned {len(ocr_text)} chars", flush=True)
                    # Only adopt OCR output if it actually returned something
                    # meaningful; on a page that's truly blank both paths return
                    # nothing and we keep the (empty) native output.
                    if len(ocr_text) >= len(text):
                        text = ocr_text
                except Exception as e:
                    # Logged loudly so we can diagnose Tesseract install issues
                    # — the previous silent `pass` here was the reason scanned
                    # PDFs kept hitting the upstream "scanned image" error
                    # without us seeing the underlying cause.
                    print(f"[OCR] page {page_num}: FAILED — {type(e).__name__}: {e}", flush=True)

            pages.append(f"--- PAGE {page_num} ---\n{text}")
    finally:
        doc.close()

    return "\n\n".join(pages)


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

# ---------------------------------------------------------------------------
# Tier 2b — Font-scan in-page heading detector (experimental, flag-gated)
# ---------------------------------------------------------------------------
#
# Motivation (thesis §3.7):
#   `doc.get_toc()` returns the PDF's EMBEDDED table of contents. Many
#   textbooks ship with a stripped TOC: chapter starts only, no section or
#   sub-section entries. For a thesis-grade syllabus-to-resource matcher
#   we want section-level page pointers ("5.3 Scheduling Algorithms, p. 266")
#   not just chapter-level ones ("Chapter 5: CPU Scheduling, p. 261").
#
#   The only signal a PDF keeps when sub-section entries are missing is
#   TYPOGRAPHIC — headings are rendered in a larger or bolder font than body
#   text. PyMuPDF exposes this via page.get_text("dict"), which returns every
#   text span with its font size and bold flag. A two-pass scan finds the
#   modal body font size, then promotes any short line set in a larger font
#   (or bold) to a heading candidate.
#
# Cost:
#   O(pages) with page.get_text("dict") ~ 30-50 ms per page on a digital PDF.
#   A 1000-page textbook takes ~30-50 s. This is why the scan is flag-gated
#   and defaults OFF — fine for admin re-index calls, too slow for the hot
#   upload path unless you explicitly opt in.
#
# Thesis framing (chapter-source ladder, §3.6):
#   Tier 1   : manual entry by uploader                  — highest trust
#   Tier 2a  : doc.get_toc()                             — exact, accurate
#   Tier 2b  : FONT-SCAN IN-PAGE HEADINGS  (← this)      — new: fills gaps
#                                                          when TOC is shallow
#   Tier 3   : Groq estimate from book metadata          — informed guess
#
# Limitations (explicit in thesis §3.7):
#   - Heuristic-only. Bold-emphasized inline text can leak through.
#   - Font flag bit 16 = bold is publisher-dependent; some PDFs use separate
#     bold faces instead of the flag, which this scan misses.
#   - Completely useless on image-only (scanned) PDFs — no font info at all.

USE_PAGE_LEVEL_HEADING_SCAN = False   # default off — re-index endpoint flips it per-call

# Running-head / footer detector: any line repeating on more than this many
# pages is treated as a chrome element (page header, chapter name in margin)
# and dropped. 5 is empirical — low enough to drop running heads that repeat
# every page, high enough to keep a sub-heading that happens to appear in 3
# different chapters' intros.
_RUNNING_HEAD_THRESHOLD_PAGES = 5


def extract_in_page_headings_from_bytes(
    pdf_bytes: bytes,
    max_headings: int = 300,
) -> list[dict]:
    """
    Scan every page for text lines set in a larger-than-body font and emit
    them as heading candidates. Output is deduplicated and running-heads
    are filtered out.

    Returns:
      [{"title": str, "page": int, "detected_by": "font_scan"}, ...]

    Runs TWO passes over the document:
      Pass 1 — find modal body font size across the first 30 content pages.
               "Modal" is more robust than mean: a few large-font front-matter
               pages would otherwise drag the mean up and suppress real headings.
      Pass 2 — collect short text lines whose max-span font size exceeds
               body_size + 1.0, OR that are bold and exceed body_size + 0.5.

    Why max-span (not mean-span) per line: a line like "5.3.2 Priority
    Scheduling" often has the leading number in a slightly larger display
    font than the trailing words. Max catches that line as a heading.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []

    # --- Pass 1: find modal body font size ---------------------------------
    size_counts: dict[int, int] = {}
    scan_pages = min(30, len(doc))
    for p in range(scan_pages):
        try:
            td = doc[p].get_text("dict")
        except Exception:
            continue
        for block in td.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sz = round(span.get("size", 0))
                    if 8 <= sz <= 20:
                        size_counts[sz] = size_counts.get(sz, 0) + 1

    if not size_counts:
        doc.close()
        return []

    body_size = max(size_counts, key=size_counts.get)
    heading_threshold = body_size + 1.0
    bold_threshold    = body_size + 0.5

    # --- Pass 2: collect heading candidates --------------------------------
    candidates: list[dict] = []
    seen_counts: dict[str, int] = {}   # normalised title -> # pages it appears on

    for p in range(len(doc)):
        try:
            td = doc[p].get_text("dict")
        except Exception:
            continue
        page_num = p + 1
        page_seen_on_this_page: set[str] = set()

        for block in td.get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                max_sz  = max((s.get("size", 0) for s in spans), default=0)
                any_bold = any((s.get("flags", 0) & 16) for s in spans)  # bit 16 = bold
                text = " ".join(s.get("text", "") for s in spans).strip()

                # --- Length & character sanity ---------------------------
                if not (2 <= len(text) <= 80):
                    continue
                if text.isdigit():
                    continue
                alpha_count = sum(1 for c in text if c.isalpha())
                if alpha_count < 2:
                    continue

                # --- Font signal ----------------------------------------
                is_heading = (
                    max_sz > heading_threshold
                    or (any_bold and max_sz > bold_threshold)
                )
                if not is_heading:
                    continue

                # --- Sentence-text rejection (the Galvin-preface bug) ---
                # Galvin's preface and author-bio use a slightly inflated
                # body font — every line of prose passed the font filter.
                # Real headings aren't sentence fragments. These rules
                # drop paragraph text without hurting true sub-section
                # headings like "5.3 Scheduling Algorithms".
                last_char = text[-1]
                if last_char in "-,;:":
                    # hyphenated line break, mid-list, mid-clause
                    continue
                # mid-sentence punctuation: period-space-lowercase is a
                # sentence boundary inside running text, never a heading
                if ". " in text:
                    idx = text.find(". ")
                    if idx + 2 < len(text) and text[idx + 2].islower():
                        continue
                # headings are short — prose paragraphs are not
                if len(text.split()) > 12:
                    continue
                # headings start with uppercase, a digit (e.g. "5.1"),
                # or a keyword prefix — never with a lowercase word
                # (which means the line is a mid-paragraph continuation)
                first_char = text[0]
                if not (first_char.isupper() or first_char.isdigit()):
                    continue

                norm = text.lower().strip()
                candidates.append({"title": text, "page": page_num, "norm": norm})
                # Count unique pages (not total occurrences) — catches running
                # heads that appear 1× per page across many pages.
                if norm not in page_seen_on_this_page:
                    seen_counts[norm] = seen_counts.get(norm, 0) + 1
                    page_seen_on_this_page.add(norm)

    doc.close()

    # --- Filter running heads & dedupe -------------------------------------
    output: list[dict] = []
    dedup_keys: set[tuple] = set()
    for c in candidates:
        if seen_counts.get(c["norm"], 0) > _RUNNING_HEAD_THRESHOLD_PAGES:
            continue
        key = (c["norm"], c["page"])
        if key in dedup_keys:
            continue
        dedup_keys.add(key)
        output.append({
            "title": c["title"],
            "page":  c["page"],
            "detected_by": "font_scan",
        })
        if len(output) >= max_headings:
            break

    return output


def extract_chapter_headings_from_bytes(
    pdf_bytes: bytes,
    *,
    include_page_scan: bool | None = None,
) -> list[dict]:
    """
    Extract chapter headings WITH page numbers from a book PDF.

    Returns a list of dicts: [{"title": "Dynamic Programming", "page": 89}, ...]

    Strategy (priority order — matches the 3-tier thesis ladder):
      1. PRIMARY (Tier 2a): doc.get_toc() — embedded table of contents.
         Exact titles + exact page numbers. Works on any digitally-created PDF.

      2. FALLBACK (pre-Tier 2a): heuristic text-block scan of first 30 pages.
         Used when get_toc() returns empty (scanned PDFs, no embedded TOC).
         Returns {"title": heading, "page": null} — page unavailable.

      3. OPTIONAL SUPPLEMENT (Tier 2b, experimental):
         If `include_page_scan=True` (or the module flag
         USE_PAGE_LEVEL_HEADING_SCAN is on), additionally run the font-based
         in-page heading scanner and MERGE its results with whatever Tier 2a
         / fallback returned. Dedupe against existing entries by
         (lowercased-title, page) so TOC entries always win over font-scan
         duplicates of the same heading.

    Why merge rather than replace:
      The embedded TOC, when present, is the authoritative source — a
      publisher's own metadata. Font-scan is a heuristic. Merging lets the
      matcher use TOC first (strong signal) and fill the sub-chapter gaps
      from font-scan (weaker but broader coverage). This is the LlamaRec-
      style "don't replace high-precision signal with heuristic; combine
      them" pattern (thesis §3.7).

    Args:
      pdf_bytes         the PDF file as bytes
      include_page_scan explicit override. None = fall back to module flag.
                        True / False force on / off for this call.
    """
    if include_page_scan is None:
        include_page_scan = USE_PAGE_LEVEL_HEADING_SCAN

    headings: list[dict] = []
    seen_title: set[str] = set()

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return []

    # --- Tier 2a: embedded TOC ---------------------------------------------
    toc = doc.get_toc()   # [[level, title, page], ...]
    if toc:
        for _level, title, page in toc:
            title = title.strip()
            if title and title not in seen_title:
                headings.append({"title": title, "page": page})
                seen_title.add(title)
            if len(headings) >= 50:
                break
    else:
        # --- Fallback: heuristic text-block scan (pre-existing behaviour) ---
        for page_num, page in enumerate(doc):
            if page_num > 30:
                break
            try:
                blocks = page.get_text("blocks")
            except Exception:
                continue
            for block in blocks:
                text = block[4].strip() if len(block) > 4 else ""
                if not text or len(text) > 100:
                    continue
                is_chapter = (
                    text.lower().startswith(("chapter", "unit", "section", "part", "module"))
                    or text[:2].isdigit()
                    or (text[0].isupper() and len(text.split()) <= 8)
                )
                if is_chapter and text not in seen_title:
                    headings.append({"title": text, "page": None})
                    seen_title.add(text)
                if len(headings) >= 30:
                    break

    doc.close()

    # --- Tier 2b (optional): merge font-scanned in-page headings -----------
    if include_page_scan:
        in_page = extract_in_page_headings_from_bytes(pdf_bytes)
        existing_keys = {
            (h["title"].lower().strip(), h.get("page"))
            for h in headings
        }
        for ip in in_page:
            key = (ip["title"].lower().strip(), ip["page"])
            if key in existing_keys:
                continue
            headings.append({"title": ip["title"], "page": ip["page"]})
            existing_keys.add(key)
            # Higher cap when page-scan is on — a real textbook can legit
            # produce 150-250 distinct headings once sub-sections are included.
            if len(headings) >= 300:
                break

    return headings


# ---------------------------------------------------------------------------
# STEP 3b — LLM-judged match (two-stage retrieval, thesis §3.6)
# ---------------------------------------------------------------------------
#
# Motivation (thesis §3.6):
#   Pure cosine over Sentence-BERT vectors has a known failure mode on
#   professor-citation strings. When the parser extracts the title/author
#   fields noisily (author-as-title, publisher bleeding into author, etc.),
#   a cosine score of 0.32 between "Peter Baer Galvin" and "Operating System
#   Concepts" rejects a book any human recognises as the same work.
#
# Solution:
#   Adopt the two-stage retrieval pattern from LlamaRec (He et al., 2023):
#     Stage 1: embedding-based shortlist (cosine over all vectors → top-K)
#     Stage 2: LLM-based disambiguation (Groq reads both sides in natural
#              language, picks the right one, explains why — or says "none")
#
#   Stage 2 rescues matches that share author tokens or publisher-independent
#   title words but score low on raw embedding distance, and it returns "null"
#   when genuinely none of the candidates is the same work — which is the
#   correct behaviour we could not get from a cosine threshold.
#
# Cost: ~1 Groq call per use (300 tokens in, 50 out). Latency ~400-600 ms.

# Search-relevance filter prompt (different shape from BOOK_JUDGE_PROMPT —
# this one filters a list rather than picking one item). Used by
# llm_filter_search_results to drop semantically-irrelevant cosine matches
# from the search bar and the agent's semantic_search tool.
SEARCH_RELEVANCE_PROMPT = """A user searched our book library for: "{query}"

Cosine similarity returned these candidates:
{candidates_block}

For EACH candidate, decide whether it is genuinely relevant to what the user asked for.

Rules (apply in this order):
1. DIRECT TITLE MATCH — if the candidate's title contains the query words (or the query is a clear partial title — "silent patient" → "The Silent Patient", "harry potter" → "Harry Potter and the Goblet of Fire"), it IS relevant. Keep it. Do not second-guess obvious title matches.
2. AUTHOR MATCH — if the query names an author and the candidate is by that author, keep it.
3. GENRE / SUBJECT MATCH — if the query describes a genre or subject ("thrillers", "machine learning", "operating systems") and the candidate is in that genre/subject, keep it.
4. INCIDENTAL KEYWORD OVERLAP — if the only relationship is that the query and the candidate share generic words ("system", "concepts", "guide", "introduction") but differ in subject, REJECT. Example: query "Harry Potter" should NOT keep "Operating System Concepts" merely because both are book titles with capitalised words.
5. When uncertain after applying rules 1–4, lean toward keeping. False rejections (a relevant book hidden) annoy users more than false positives.

Return ONLY valid JSON (no markdown, no code fences):
{{
  "relevant": [list of candidate numbers (1-{n}) that are genuinely relevant],
  "reason": "one short sentence — why the rejected candidates were rejected, OR empty string if all were kept"
}}

If NO candidates are relevant, return relevant: []."""


def llm_filter_search_results(query: str, candidates: list[dict]) -> list[int]:
    """
    Stage-2 relevance filter for semantic-search retrieval surfaces.

    Args:
      query       The user's natural-language search query
      candidates  Cosine-ranked top-K from search_books, each with at least
                  "title" and "author" fields.

    Returns:
      List of 0-based indices into `candidates` that the LLM judges
      genuinely relevant to the query. Empty list = drop everything.

    Failure mode: if the Groq call or JSON parse fails, returns ALL indices
    (i.e. no filtering applied) on the principle that an LLM hiccup should
    not silently empty the results page. The cosine floor in the caller is
    the safety net.

    Cost: ~1 Groq call per search (~250 tokens in, ~40 out). Latency
    ~300-500ms. Worth it on the search bar (one call per user query) and
    on the agent's semantic_search tool (called at most twice per turn).
    """
    if not candidates:
        return []

    n = len(candidates)
    lines = []
    for i, c in enumerate(candidates, start=1):
        title  = (c.get("title") or "").strip() or "<untitled>"
        author = (c.get("author") or c.get("authors") or "").strip() or "unknown"
        lines.append(f"  {i}. {title} — {author}")

    prompt = SEARCH_RELEVANCE_PROMPT.format(
        query=query.strip(),
        candidates_block="\n".join(lines),
        n=n,
    )

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        keep = data.get("relevant", [])
        # 1-based → 0-based, validate range
        indices = [int(k) - 1 for k in keep if isinstance(k, (int, str))]
        return [i for i in indices if 0 <= i < n]
    except Exception:
        # On failure, fall back to "keep everything" — the cosine floor
        # in the caller has already done a first-pass safety check.
        return list(range(n))


BOOK_JUDGE_PROMPT = """You are matching a professor's recommended textbook against a library catalogue.

Professor recommended:
  "{query}"

Candidates from our library (ranked by semantic similarity, top-{n}):
{candidates_block}

Which candidate is the SAME BOOK as what the professor recommended?

Rules:
- Different editions or different publishers of the same book count as SAME.
- A PDF soft-copy and a physical listing of the same textbook count as SAME.
- Author-name overlap is a strong signal even if title words differ slightly.
- If NONE of the candidates is the same book, return match: null.

Return ONLY valid JSON (no markdown, no code fences):
{{
  "match": <candidate number 1-{n}, or null>,
  "confidence": "high" | "medium" | "low",
  "reason": "one short sentence — which author/title markers confirmed the match"
}}"""


CHAPTER_JUDGE_PROMPT = """You are picking the best chapter of a textbook for a specific lecture-plan unit.

Lecture unit: "{query}"

Candidate chapters (ranked by semantic similarity, top-{n}):
{candidates_block}

Which chapter best covers this lecture unit?

Rules:
- Prefer a chapter whose title directly names the unit's core topic.
- Chapters that only brush the topic in passing do NOT count.
- If NO chapter is a strong match, return match: null.

Return ONLY valid JSON (no markdown, no code fences):
{{
  "match": <candidate number 1-{n}, or null>,
  "confidence": "high" | "medium" | "low",
  "reason": "one short sentence"
}}"""


def llm_judge_match(
    query: str,
    candidates: list[dict],
    kind: str,
) -> dict:
    """
    Stage-2 disambiguation over a cosine-shortlisted candidate set.

    Args:
      query       Raw query text (e.g. professor citation "Silberschatz,
                  Operating System Concepts, Wiley" OR a lecture unit title
                  "CPU Scheduling").
      candidates  Shortlist of top-K items from cosine stage.
                  For kind="book":    [{"title": str, "authors": str}, ...]
                  For kind="chapter": [{"title": str, "page": int | None}, ...]
      kind        "book" | "chapter"

    Returns:
      {
        "match_index": int | None,   # 0-based into `candidates`. None = no match.
        "confidence":  "high" | "medium" | "low",
        "reason":      str,          # why the LLM picked (or rejected) — auditable
      }

    Error handling:
      Any Groq failure, JSON-parse failure, or out-of-range index falls back to
      match_index=0 with confidence="low". The endpoint never crashes on LLM
      issues; the downstream caller decides whether "low"-confidence matches
      are acceptable (today: book matcher accepts low, chapter matcher does not).
    """
    if not candidates:
        return {"match_index": None, "confidence": "low", "reason": "empty shortlist"}

    n = len(candidates)

    if kind == "book":
        lines = []
        for i, c in enumerate(candidates, start=1):
            title   = (c.get("title") or "").strip() or "<untitled>"
            authors = (c.get("authors") or c.get("author") or "").strip() or "unknown"
            lines.append(f"  {i}. {title} — {authors}")
        prompt = BOOK_JUDGE_PROMPT.format(
            query=query.strip(),
            candidates_block="\n".join(lines),
            n=n,
        )
    elif kind == "chapter":
        lines = []
        for i, c in enumerate(candidates, start=1):
            title = (c.get("title") or "").strip() or "<untitled>"
            page  = c.get("page")
            if page:
                lines.append(f"  {i}. {title} (page {page})")
            else:
                lines.append(f"  {i}. {title}")
        prompt = CHAPTER_JUDGE_PROMPT.format(
            query=query.strip(),
            candidates_block="\n".join(lines),
            n=n,
        )
    else:
        raise ValueError(f"llm_judge_match: unknown kind {kind!r}")

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,      # reproducibility — same query+candidates ⇒ same pick
            max_tokens=150,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())

        m = data.get("match")
        if m is None:
            return {
                "match_index": None,
                "confidence":  data.get("confidence", "low"),
                "reason":      (data.get("reason") or "").strip(),
            }
        idx = int(m) - 1   # LLM emits 1-based; convert to 0-based
        if idx < 0 or idx >= n:
            return {
                "match_index": None,
                "confidence":  "low",
                "reason":      f"LLM returned out-of-range index {m}",
            }
        return {
            "match_index": idx,
            "confidence":  data.get("confidence", "medium"),
            "reason":      (data.get("reason") or "").strip(),
        }
    except Exception as e:
        # Never let a Groq hiccup kill the endpoint — fall back to cosine top-1
        # with low-confidence flag so the caller can decide to trust it or not.
        return {
            "match_index": 0,
            "confidence":  "low",
            "reason":      f"LLM judge failed ({type(e).__name__}); fell back to cosine top-1",
        }


# ---------------------------------------------------------------------------
# STEP 4 — Find best chapter match for a topic inside a specific book
# ---------------------------------------------------------------------------

def find_best_chapter(topic_vec: np.ndarray, book: dict, *, unit_title: str = "") -> dict:
    """
    Two-stage chapter match for a lecture unit inside one book.

    Stage 1 — cosine shortlist:
      Score every chapter heading by cosine(topic_vec, embed(heading_title)),
      take the top-5.

    Stage 2 — LLM judge (when `unit_title` is provided):
      Ask Groq which of the 5 shortlisted chapters actually covers the unit,
      or "none". The LLM can see chapter titles + page numbers and the unit
      title verbatim — better at recognising e.g. "Chapter 5: CPU Scheduling"
      as the right answer for unit "Process Scheduling" even if cosine ranked
      it #3 behind two incidental matches.

    Returns:
      {
        "title":       str,          # "" if no match
        "page":        int | None,
        "score":       float,        # cosine score of chosen chapter
        "match_reason": str          # why the LLM picked it (auditable)
      }

    Backward compatibility:
      Legacy callers that don't pass `unit_title` get the old cosine-only
      argmax behaviour — no Groq call, no reason field change.
    """
    headings = book.get("chapter_headings", [])
    if not headings:
        return {"title": "", "page": None, "score": 0.0, "match_reason": "no chapter headings on book"}

    # Stage 1: score every chapter
    scored: list[tuple[float, dict]] = []
    for h in headings:
        # support both old (plain string) and new (dict) heading formats
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
        scored.append((h_score, {"title": h_title, "page": h_page}))

    if not scored:
        return {"title": "", "page": None, "score": 0.0, "match_reason": "all headings empty"}

    scored.sort(key=lambda x: x[0], reverse=True)

    # Legacy path — caller did not pass unit_title (no Groq call)
    if not unit_title:
        top_score, top = scored[0]
        return {
            "title":       top["title"],
            "page":        top["page"],
            "score":       round(top_score, 4),
            "match_reason": "cosine top-1 (legacy caller, no LLM judge)",
        }

    # Stage 2: LLM judge over top-5
    shortlist = [s[1] for s in scored[:5]]
    judgement = llm_judge_match(unit_title, shortlist, kind="chapter")

    if judgement["match_index"] is None:
        # LLM explicitly rejected all candidates — honour that rather than
        # forcing a match. Downstream Tier-3 estimator will take over.
        return {
            "title":       "",
            "page":        None,
            "score":       round(scored[0][0], 4),
            "match_reason": f"LLM judge: no strong chapter match ({judgement['reason']})",
        }

    picked      = shortlist[judgement["match_index"]]
    picked_score = scored[judgement["match_index"]][0]
    return {
        "title":       picked["title"],
        "page":        picked["page"],
        "score":       round(picked_score, 4),
        "match_reason": f"LLM judge ({judgement['confidence']}): {judgement['reason']}",
    }


# ---------------------------------------------------------------------------
# STEP 5 — Find professor-recommended books in inventory (vector search)
# ---------------------------------------------------------------------------

PROFESSOR_BOOK_MATCH_THRESHOLD = 0.60   # cosine sim threshold (legacy, cosine-only path)
BOOK_SHORTLIST_K              = 5      # top-K across books + pdfs for LLM judge stage
BOOK_COSINE_FLOOR             = 0.20   # below this, don't even ask the LLM — no hope

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

    Two-stage retrieval (thesis §3.6, follows LlamaRec, He et al. 2023):
      Stage 1 — cosine shortlist:
        Score every physical book AND every PDF against the professor's
        citation (title + authors). Take the top-K across the *merged* pool
        so physical and digital compete on the same scale.
      Stage 2 — LLM judge:
        Ask Groq which shortlisted candidate is the same book as the
        professor's recommendation, or "none". The LLM sees both sides in
        natural language — it recovers matches that cosine rejects at 0.32
        just because author tokens dominate the citation string, and it
        correctly rejects a shortlist full of near-neighbours when the
        recommended book genuinely isn't in our inventory.

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
        "match_score":     float,     # cosine of chosen candidate (auditable)
        "match_reason":    str,       # LLM's one-line justification (or rejection reason)
        "match_confidence": str       # "high" | "medium" | "low"
      }

    Note: `book_id` holds the ID regardless of source_type — downstream
    map_units_to_chapters already splits the flat `preferred_book_ids` list
    into books vs pdfs by checking each collection, so a PDF id flows through
    naturally.
    """
    books_with_vectors = [b for b in books if b.get("vector")]
    pdfs_with_vectors  = [p for p in (pdfs or []) if p.get("embedding")]
    results = []

    def _empty_result(rec_title, rec_authors, reason, score=0.0):
        return {
            "recommended_title":   rec_title,
            "recommended_authors": rec_authors,
            "found":           False,
            "source_type":     "",
            "book_title":      "",
            "book_author":     "",
            "book_id":         "",
            "cloudinary_url":  "",
            "available":       False,
            "match_score":     round(score, 4),
            "match_reason":    reason,
            "match_confidence": "low",
        }

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

        # --- Stage 1: cosine-score every candidate, merge physical + pdf ---
        scored: list[tuple[float, dict, str]] = []  # (score, item_dict, type)
        for book in books_with_vectors:
            book_vec = np.array([book["vector"]])
            s = float(cosine_similarity(rec_vec, book_vec)[0][0])
            scored.append((s, book, "physical"))
        for pdf in pdfs_with_vectors:
            pdf_vec = np.array([pdf["embedding"]])
            s = float(cosine_similarity(rec_vec, pdf_vec)[0][0])
            scored.append((s, pdf, "pdf"))

        if not scored:
            results.append(_empty_result(rec_title, rec_authors,
                                         "no candidates in inventory"))
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score = scored[0][0]

        # Sanity floor — if even the best cosine is 0.10, the shortlist is
        # noise and the LLM would just hallucinate a pick. Skip the Groq call.
        if top_score < BOOK_COSINE_FLOOR:
            results.append(_empty_result(
                rec_title, rec_authors,
                f"top cosine {top_score:.2f} < floor {BOOK_COSINE_FLOOR} — nothing close enough",
                score=top_score,
            ))
            continue

        shortlist = scored[:BOOK_SHORTLIST_K]
        candidate_payload = []
        for _, item, t in shortlist:
            candidate_payload.append({
                "title":   item.get("title", ""),
                "authors": (
                    item.get("author", "") if t == "physical"
                    else item.get("professor", "")
                ),
            })

        # --- Stage 2: LLM judge picks the real match (or says "none") ---
        query_str = f"{rec_title} — {rec_authors}".strip(" —")
        judgement = llm_judge_match(query_str, candidate_payload, kind="book")

        if judgement["match_index"] is None:
            results.append(_empty_result(
                rec_title, rec_authors,
                f"LLM judge: no candidate is the same book — {judgement['reason']}",
                score=top_score,
            ))
            continue

        idx = judgement["match_index"]
        picked_score, picked_item, picked_type = shortlist[idx]
        item_author = (
            picked_item.get("author", "") if picked_type == "physical"
            else picked_item.get("professor", "")
        )
        results.append({
            "recommended_title":   rec_title,
            "recommended_authors": rec_authors,
            "found":               True,
            "source_type":         picked_type,
            "book_title":          picked_item.get("title", ""),
            "book_author":         item_author,
            "book_id":             str(picked_item.get("_id", "")),
            "cloudinary_url":      picked_item.get("cloudinary_url", "") if picked_type == "pdf" else "",
            "available":           (
                picked_item.get("available", True) if picked_type == "physical"
                else True   # PDFs are always downloadable
            ),
            "match_score":         round(picked_score, 4),
            "match_reason":        judgement["reason"] or "LLM judge picked this candidate",
            "match_confidence":    judgement["confidence"],
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

            # Stage-2 LLM judge on the AI-suggested fallback (Chapter 3 §3.6).
            # Added after user testing surfaced an Object-Oriented-Programming
            # lecture plan being matched to "Operating System Concepts" purely
            # because (a) prof's actual OOP textbooks were not in the library
            # and (b) cosine argmax over the OS-heavy inventory still returned
            # a winner at 0.43 sim. The LLM judge reads the unit title and the
            # candidate book title in natural language and rejects matches
            # that share keywords but not subject. When it rejects, we drop
            # the suggestion entirely so the UI shows "no relevant book in
            # inventory" rather than a confidently-labeled wrong-domain pick.
            if chosen_book is not None:
                judgement = llm_judge_match(
                    query=unit_title,
                    candidates=[{
                        "title":   chosen_book.get("title", ""),
                        "authors": chosen_book.get("author", "") or chosen_book.get("professor", ""),
                    }],
                    kind="book",
                )
                if judgement["match_index"] is None:
                    chosen_book  = None
                    chosen_score = 0.0

        if not chosen_book:
            # Emit an empty unit-match record so the frontend can render
            # "no relevant book found in our inventory for this unit" rather
            # than silently dropping the unit (which made the OOP plan look
            # half-broken — units 5+ were missing entirely from the result).
            results.append({
                "unit_no":            unit.get("unit_no", ""),
                "unit_title":         unit_title,
                "book_title":         "",
                "book_author":        "",
                "source":             "no_match",
                "match_score":        0.0,
                "formats_available":  [],
                "suggested_chapter":  "",
                "chapter_page":       None,
                "chapter_source":     "none",
                "chapter_confidence": "low",
                "no_match_reason":    "No book in our inventory closely matches this unit's topic.",
            })
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

        chapter = find_best_chapter(topic_vec, chapter_source_book, unit_title=unit_title)

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
    department: str,     # kept in signature for future use; deliberately unused today
    pdfs: list[dict] | None,
    books: list[dict] | None = None,
    limit_per_type: int = 3,
) -> list[dict]:
    """
    Find peer-uploaded material tagged to the *exact course* of the lecture plan.

    Searches BOTH collections:
      - pdfs   → digital uploads (notes, PYQs, reference PDFs, soft-copy textbooks)
      - books  → physical listings where resource_type ∈ {notes, previous_papers,
                 reference} — a student lending hand-written notes or hard-copy PYQs

    Match precedence (course only — not department):
      1. course_code prefix:    doc.course starts with "MCC510"  — strongest
      2. course_code substring: doc.subject contains "MCC510"   — medium
      3. course_name substring: doc.course or doc.subject contains the name

    Why `startswith` for code: upload forms now store course as
    "MCC510 - Operating Systems" (the CODE - Name format from /api/courses).
    A plain substring match would also let "MCC51" hit "MCC510", "MCC511",
    "MCC512" — a real-world false positive. `startswith(code + " ")` (or
    exactly equal to code) is the safe precise match.

    Why NOT department: a single department (e.g. Computer Science) hosts 50+
    courses. Falling back to department would show DBMS, AI, ML resources on
    an OS lecture plan — same department, but useless. Students tag uploads
    with a specific course; honour that.

    Returns flat list capped per resource_type. Each item carries:
      source_type ∈ {"pdf", "physical"}  — lets the Node view render
        /pdfs/:id vs /books/:id links without a second DB lookup.
    """
    course_name_lc = (course_name or "").strip().lower()
    course_code_lc = (course_code or "").strip().lower()

    # No course signal → return nothing. Department alone is too noisy.
    if not (course_name_lc or course_code_lc):
        return []

    # Normalised prefix used for exact-code checks: "mcc510 " — the trailing
    # space stops "mcc51" from matching "mcc510 - ...". We also accept the
    # code alone (doc.course == "mcc510") in case the uploader omitted the name.
    code_prefix = (course_code_lc + " ") if course_code_lc else ""

    def _classify(doc_course: str, doc_subject: str) -> str | None:
        """Return match_reason or None. Strongest signal wins."""
        c = (doc_course or "").strip().lower()
        s = (doc_subject or "").strip().lower()

        # 1. Exact course-code match (prefix or whole field)
        if course_code_lc and (c == course_code_lc or c.startswith(code_prefix)):
            return "course_code_exact"

        # 2. Course code appears inside subject/course body
        if course_code_lc and (course_code_lc in c or course_code_lc in s):
            return "course_code_substring"

        # 3. Course name substring match (weakest — "Operating Systems" may
        # legitimately appear in a related-but-not-identical course)
        if course_name_lc and (course_name_lc in c or course_name_lc in s):
            return "course_name"

        return None

    matches: list[dict] = []

    # --- PDFs --------------------------------------------------------------
    for pdf in (pdfs or []):
        reason = _classify(pdf.get("course", ""), pdf.get("subject", ""))
        if not reason:
            continue
        matches.append({
            "source_type":    "pdf",
            "pdf_id":         str(pdf.get("_id", "")),
            "title":          pdf.get("title", ""),
            "subject":        pdf.get("subject", ""),
            "course":         pdf.get("course", ""),
            "department":     pdf.get("department", ""),
            "professor":      pdf.get("professor", ""),
            "resource_type":  pdf.get("resource_type", "notes"),
            "cloudinary_url": pdf.get("cloudinary_url", ""),
            "match_reason":   reason,
        })

    # --- Physical books tagged as notes/PYQs/reference ---------------------
    # Skip plain textbooks: those are handled by find_recommended_books_in_inventory
    # (matched against professor's textbook list) and by unit-level semantic
    # matching. Here we only want the loose physical study material.
    for book in (books or []):
        rt = (book.get("resource_type") or "textbook").lower()
        if rt not in {"notes", "previous_papers", "reference"}:
            continue
        # Books don't have a "subject" field — use description as a soft
        # fallback signal so a note titled "OS revision" with description
        # "MCC510 mid-sem prep" still matches.
        reason = _classify(book.get("course", ""), book.get("description", ""))
        if not reason:
            continue
        matches.append({
            "source_type":   "physical",
            "book_id":       str(book.get("_id", "")),
            "title":         book.get("title", ""),
            "author":        book.get("author", ""),
            "course":        book.get("course", ""),
            "department":    book.get("department", ""),
            "resource_type": rt,
            "available":     book.get("available", True),
            "match_reason":  reason,
        })

    # Cap per resource_type so one flooded category doesn't crowd others.
    # Rank within a type by match_reason strength so exact-code hits win.
    reason_rank = {"course_code_exact": 0, "course_code_substring": 1, "course_name": 2}
    by_type: dict[str, list[dict]] = {}
    for m in matches:
        by_type.setdefault(m["resource_type"], []).append(m)

    trimmed: list[dict] = []
    for items in by_type.values():
        items.sort(key=lambda x: reason_rank.get(x["match_reason"], 99))
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
        # Hard PyMuPDF failure (corrupt / encrypted file)
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

    # --- Scanned-PDF detection ---------------------------------------------
    # extract_text_from_bytes always emits per-page "--- PAGE N ---" markers,
    # so a scanned image-only PDF still yields a non-empty raw_text that has
    # near-zero actual content. Strip the markers and count characters of
    # real text. Below the floor (50 chars across the whole document) we
    # bail out with a specific, user-facing error rather than letting the
    # call fall through to Groq, which will then return empty JSON and
    # surface the generic "could not parse" message that confused the
    # tester. Distinguishing scanned-image from genuinely-garbled text
    # also makes the limitation honest to document in §6.3.
    content_only = "\n".join(
        line for line in raw_text.splitlines()
        if line.strip() and not line.strip().startswith("--- PAGE")
    )
    if len(content_only.strip()) < 50:
        scanned_msg = (
            "This PDF contains no selectable text — it appears to be a "
            "scanned image, not a text-based document. BookSwap can read "
            "text-based PDFs (the original file from the professor) but "
            "cannot OCR scanned pages yet. Please upload a non-scanned "
            "version of the lecture plan."
        )
        log_curriculum_run(
            prompt_version=active_version,
            pdf_filename=pdf_filename,
            raw_text_len=len(content_only),
            parsed={},
            recommended_books=[],
            unit_matches=[],
            error=scanned_msg,
        )
        return {"error": scanned_msg, "prompt_version": active_version}

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
        books=books,    # include physical notes / PYQs / reference material
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

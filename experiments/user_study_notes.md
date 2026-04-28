# User study — raw session notes

This file is a chronological dump of observations from each session of
the BookSwap user study. It is the source-of-truth artefact that the
thesis §5.7 narrative cites; the session-aggregated tables in §5.7.2 –
§5.7.5 are derived from this file.

Each session block follows the same shape:
- header line with date, code, dept/year
- pre-task answers
- per-task observations (verbatim quotes wherever possible)
- closing remarks
- code fixes triggered by the session, with commit hashes

---

## Pilot session — 2026-04-28

**Participant.** Mechanical Engineering, year [redacted to keep
identifiable details out of a public repo]. First name Swathi (used
only in oral debrief; not stored elsewhere). Treated as a pilot, not
part of the n = 8 formal cohort.

**Build under test.** Commit `8c585b9` (bookswap) + `41bd5ea`
(bookswap-ai). Pre-pilot. The fixes listed at the bottom of this block
were applied before the formal cohort began.

### Pre-task answers
- Department: Mechanical Engineering
- Year: omitted from notes
- Q3 (1–5 ease of finding study material today): omitted

### Session observations

**Register / login.** Smooth. ~1–2 sec end-to-end. No friction
observed.

**PDF upload (intended Task 2 in the formal protocol).** Took ~2 min
end-to-end. Two distinct verbatim observations:

> *"no need of separate pdf upload route — only post-a-book is enough"*

> *"no need of subject in it"*

The participant entered a single upload but two records appeared in
the database afterwards. The cause was not pinned down during the
session. Hypothesis: form double-submit; the unified upload form on
`/books/new` POSTs to `/books`, while the legacy `/pdfs/new` form
POSTs to `/pdfs`, and a momentary network blip during the first POST
may have led the user to retry through the second route. Action item
for the formal cohort: instrument the form with a submit-once guard.

**Chapter / topic extraction (visible on the PDF show page after
upload).** The chapter list showed:

> *"📄 B / 📄 Solid / 📄 P / 📄 O x / 📄 F F / 📄 Temperature, C ∞ / 📄 Viscosity, (Ns/m m 2) / 📄 Carbon dioxide / 📄 Methane / 📄 Helium Air / 📄 Mercury / 📄 Water / 📄 8 / 📄 2 / 📄 4"*

These are not chapter titles — they are cells from a viscosity
table inside the textbook. The pilot PDF had no embedded TOC, so
`doc.get_toc()` returned empty and the heuristic block-scan
fallback triggered. The fallback's third disjunct — "starts with
uppercase, ≤ 8 words" — accepted single-character cells and short
table headers as headings.

**Search bar — query "inspirational books"**. Results returned were
*The Midnight Library*, *Verity*, *Introduction to Probability*,
*Artificial Intelligence: A Modern Approach*, and *Project Hail Mary*.
*Atomic Habits* — the canonical "inspirational" book in the inventory
— did not appear. Pure cosine similarity over MiniLM embeddings is
sensitive to vocabulary drift between the query and the book's
description; the word *inspirational* does not strongly correlate with
*habits / productivity / behaviour-change* in the embedding space, so
*Atomic Habits* did not enter the top-10 cosine candidates. Discussed
in §5.7.5 theme 4 and §6.3 limitations.

**PDF library filter — department CSE**. Returned "no resources
found." Not a bug — the inventory at the time of pilot had no CSE
resources. The participant did then filter by *MC* (Mathematics and
Computing) and successfully found the OS Concepts textbook tagged
under that department.

**Curriculum upload.** Took ~20 sec on the deployed instance. This
is dominated by Render free-tier cold-start (10–30 sec to wake the
container) plus the lazy SentenceTransformer load on the first
embedding call. Documented in §6.3.

**Curriculum results — same book missed when uploaded under
different course name.** A textbook present in the inventory tagged
under one course-name string was not surfaced for a lecture plan
that referenced the same course under its catalogue code. This is
the precision-vs-recall trade-off in `find_course_related_notes`
(see §6.3 limitations bullet on course-tag rigidity).

**Chat widget.** The participant saw messages they had not typed:

> *"random msg being displayed at the top which I didn't type"*

These were leaked from a previous user's localStorage history (the
shared laptop had been used by someone else, the chat widget rehydrated
the prior session). A subsequent send attempt returned 400 Bad Request,
which on inspection appeared to be a downstream effect of the leaked
state (the input was empty at the time the request fired; the client-
side empty-string guard handled subsequent attempts cleanly once the
state was reset). The 400 did not recur after the namespacing fix.

**Agent chat responses rating.** Not collected — the chat was unusable
for most of the session due to the localStorage leak.

**Post-upload book download behaviour.** The participant looked for a
"download PDF" affordance on the book detail page after uploading and
did not find one. Expected behaviour: physical book listings do not
have a downloadable PDF; the digital version lives on the Library
page under a separate model (`pdfs` collection). The unified upload
form on `/books/new` already supports both physical and PDF/digital
listings via a tab toggle, but the participant did not realise the
toggle existed. Improving the toggle's discoverability is part of the
follow-up addressed by the *"remove separate pdf upload route"* fix.

### Code fixes triggered by this session

| Issue | Commit | File(s) |
|---|---|---|
| Heuristic chapter extraction accepted single-character cells | _fill_with_actual_hash_ | `bookswap-ai/app/chapter_extractor.py` |
| Chat widget localStorage leaked across users | _fill_with_actual_hash_ | `bookswap/views/partials/chat-widget.ejs` |
| "Upload PDF" button on Library page was a confusing second entry point | _fill_with_actual_hash_ | `bookswap/views/pdfs/index.ejs` |
| Subject field redundant on PDF upload form | _fill_with_actual_hash_ | `bookswap/views/pdfs/new.ejs` + `bookswap/app.js` |

(Actual hashes filled in once the commits land on `main` / `master`.)

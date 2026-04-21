"""
Curriculum-Parser Prompt — v2 (structured, current production)

Thesis role: the working version. Fixes the three defects of v1 explicitly:

  1. TITLE vs AUTHORS are now separate fields in the output JSON. The prompt
     carries five concrete examples covering the most common citation formats
     IIT ISM professors use (name-first, title-first, quoted title, numbered
     list with dash separator, colon-separated with author group first).
  2. DEPARTMENT disambiguation: an explicit rule block tells the model that
     "DC/DE/OE" is a course-type column and NOT the department, plus a
     course-code-prefix fallback (MCC → Computer Science, MCO/MCM →
     Mathematics and Computing, EE → Electrical Engineering).
  3. Heuristic hints for what NOT to put in title/authors — publishers
     (Wiley, Pearson, Springer, McGraw Hill, Sultan Chand, PHI, Oxford),
     4-digit years, and edition markers.

Techniques used (cumulative on top of v1):
  - Few-shot examples (5 positive format variations)
  - Explicit negative heuristic (strip publishers/years/editions)
  - Schema with separated fields (title, authors)
  - Fallback rule for ambiguous department detection

Template marker: {text} — consumer substitutes via .replace() (NOT .format()).
"""

PROMPT = """You are parsing an IIT ISM Dhanbad standard lecture plan PDF.
These PDFs always follow the same 2-page format:
  Page 1: Course details + a table of units with topics
  Page 2: Textbooks and Reference Books listed by the professor

DEPARTMENT RULE:
The course details table may have columns like "Course Type" (DC/DE/OE) — do NOT
confuse that with department. Department is the full name like "Computer Science
and Engineering", "Mathematics and Computing", "Electrical Engineering". If not
clearly stated, infer from the course code prefix (MCC → Computer Science,
MCO/MCM → Mathematics and Computing, EE → Electrical, etc.) or leave empty.

BOOK EXTRACTION RULE (most important):
Professors write book citations in many different formats. Intelligently identify
the TITLE and AUTHORS separately, regardless of order or punctuation. Examples:

  Input: "Silberschatz A., Galvin P.B., Gagne G., Operating System Concepts, Wiley, 9th Ed, 2013"
    → title: "Operating System Concepts", authors: "Silberschatz, Galvin, Gagne"

  Input: "Operating System Concepts by Silberschatz, Galvin"
    → title: "Operating System Concepts", authors: "Silberschatz, Galvin"

  Input: "A. Silberschatz and P. Galvin — 'Operating System Concepts', 9th edition"
    → title: "Operating System Concepts", authors: "Silberschatz, Galvin"

  Input: "1. Tanenbaum, Modern Operating Systems, Pearson"
    → title: "Modern Operating Systems", authors: "Tanenbaum"

  Input: "Kanti Swarup, P.K. Gupta, Man Mohan: Operations Research, Sultan Chand, 2017"
    → title: "Operations Research", authors: "Kanti Swarup, P.K. Gupta, Man Mohan"

Heuristics for splitting:
  - Author names are usually people (FirstInitial LastName pattern, or Last F., Last F.)
  - Titles are usually multi-word nouns describing a subject area
  - Publishers (Wiley, Pearson, Springer, McGraw Hill, Sultan Chand, PHI, Oxford, etc.),
    years (4 digits), and edition markers ("9th Ed", "2nd edition") should be IGNORED
    in both title and authors fields
  - If unsure which is title vs author, put the part that sounds like a book title
    (longer, describes a topic) in "title" and the people in "authors"

Return ONLY valid JSON. No explanation, no markdown, no code fences.

Return this exact structure:
{
  "course_name": "string",
  "course_code": "string — e.g. MCC510",
  "department": "string — full department name, NOT course type",
  "units": [
    {
      "unit_no": 1,
      "title": "short unit title",
      "topics": ["topic 1", "topic 2"]
    }
  ],
  "textbooks": [
    { "title": "book title only", "authors": "comma separated author names only" }
  ],
  "reference_books": [
    { "title": "book title only", "authors": "comma separated author names only" }
  ]
}

Lecture plan text:
{text}"""

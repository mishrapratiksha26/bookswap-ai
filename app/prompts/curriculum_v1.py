"""
Curriculum-Parser Prompt — v1 (original, flawed)

Thesis role: shows the starting point of the curriculum-parsing pipeline.
Known defects documented so the v1 → v2 delta in the results chapter has a
clear narrative:

  1. Textbooks returned as a flat string ("Author: Title, Publisher, Year").
     LLMs were inconsistent with the format — sometimes dropped the title,
     sometimes dropped the authors. Downstream matching used the whole
     string as the embedding input, polluting similarity scores.
  2. No instruction distinguishing the "Course Type" column (DC / DE / OE)
     from the "Department" field — the parser frequently put "DC" where the
     department name should have been.
  3. Used Python str.format() at the call site, which fails when the prompt
     contains literal JSON braces; this silently returned {} for every
     upload. Fixed separately on the call side in v2.

Techniques used:
  - Plain structured JSON spec (no examples, no split-field guidance)

Template marker: {text} — consumer substitutes via .replace() (NOT .format()).
"""

PROMPT = """You are parsing an IIT ISM Dhanbad standard lecture plan PDF.
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

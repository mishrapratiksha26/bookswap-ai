from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from app.search import get_similar_books, search_books, get_personal_recommendations, rerank_books
from app.embeddings import generate_embedding
from pymongo import MongoClient, ASCENDING
import os
import json
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

router = APIRouter()

client = MongoClient(os.environ.get("DB_URL"))
db = client[os.environ.get("DB_NAME", "books")]
books_collection = db["books"]

# ---------------------------------------------------------------------------
# Retrieval quality floors (Chapter 3 §3.3 — Two-stage retrieval, LlamaRec-style)
#
# Every retrieval surface in BookSwap (semantic_search tool, /search endpoint,
# curriculum unit-fallback) checks that the top cosine candidate clears a
# minimum similarity before returning anything. Below the floor we return an
# empty result rather than the least-dissimilar noise.
#
# Empirically tuned after user-test feedback ("Harry Potter showed Operating
# Systems books"): pure cosine top-K against an inventory with no thrillers
# returns OS textbooks at 0.15-0.20 sim — technically the best match, but the
# best of nothing. SEARCH_COSINE_FLOOR = 0.30 keeps clearly relevant matches
# while filtering out-of-distribution queries.
# ---------------------------------------------------------------------------
SEARCH_COSINE_FLOOR  = 0.30   # below this → return empty
SEARCH_COSINE_BYPASS = 0.70   # above this → cosine is conclusive, skip LLM filter
                              # (keeps clear title matches like "silent patient"
                              # → "The Silent Patient" from being rejected by an
                              # over-cautious relevance prompt)
conversations_collection = db["conversations"]

# TTL index on last_active — MongoDB auto-deletes sessions idle > 30 minutes.
# This keeps the collection clean and ensures old conversation context doesn't
# bleed into new sessions. Using get_or_create pattern to avoid duplicate index error.
try:
    conversations_collection.create_index(
        [("last_active", ASCENDING)],
        expireAfterSeconds=1800,   # 30 minutes
        name="ttl_last_active"
    )
except Exception:
    pass  # index already exists — safe to ignore

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPTS — now sourced from app/prompts/{agent_v1..v4}.py.
#
# The four agent prompt versions for thesis RQ2 live in the registry at
# app/prompts/__init__.py. The constants below are thin aliases kept ONLY
# for backward compatibility with anything that still imports them by name
# from this module.
#
# Production path (GET /agent): reads the version from the request and calls
# get_agent_prompt(version) — not these constants.
# Evaluation path (POST /evaluate): same, via prompt_map below.
# ---------------------------------------------------------------------------

from app.prompts import AGENT_PROMPTS, AGENT_LATEST  # noqa: E402

PROMPT_V1 = AGENT_PROMPTS["v1"]
PROMPT_V2 = AGENT_PROMPTS["v2"]
PROMPT_V3 = AGENT_PROMPTS["v3"]
PROMPT_V4 = AGENT_PROMPTS["v4"]

# Legacy default — latest production prompt. Prefer get_agent_prompt() in new code.
SYSTEM_PROMPT = AGENT_PROMPTS[AGENT_LATEST]

# ---------------------------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class SimilarBooksRequest(BaseModel):
    book_id: str
    top_k: int = 5

class BookRating(BaseModel):
    book_id: str
    rating: float = 1.0

class PersonalRecommendationRequest(BaseModel):
    books: list[BookRating]
    top_k: int = 5

class AvailabilityRequest(BaseModel):
    book_ids: list[str]

class EmbedRequest(BaseModel):
    book_id: str

class AgentRequest(BaseModel):
    message: str
    user_id: Optional[str] = None      # MongoDB ObjectId string — who is asking
    session_id: Optional[str] = None   # UUID — which conversation this belongs to
    # Thesis RQ2: allow the Node.js layer (or an evaluation harness) to pin
    # a specific agent prompt version per-request — "v1".."v4". Unset means
    # "use the latest" (AGENT_LATEST in app/prompts/__init__.py). Every run
    # is logged to experiments/agent_runs.jsonl alongside the version used.
    prompt_version: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    session_id: str          # echoed back so the frontend can store it
    tools_called: List[str]  # ordered list: e.g. ["semantic_search", "check_availability"]
    iterations: int          # how many loop cycles ran — used in Chapter 5 evaluation
    # Books / PDFs that came back from any tool call during this turn,
    # captured so the chat widget can hyperlink any of these titles
    # appearing in the agent's free-text response.  Each item:
    #   {id: "<mongo objectid string>", title: "...", type: "book"|"pdf"}
    # User feedback from Shweta's session: "could you include links of
    # these books so when I click on the book name I can directly go
    # to borrow it" — this is the wire-side half of that fix; the
    # client-side linkifier consumes this list.
    cited_resources: List[dict] = []

# ---------------------------------------------------------------------------
# TOOL DEFINITIONS — 5 tools total.
#
# Tool 1: semantic_search      — vector search + re-ranking (Phase 4)
# Tool 2: check_availability   — real-time borrow status from DB
# Tool 3: get_alternatives     — similarity search for unavailable books
# Tool 4: get_user_profile     — user taste vector + borrow history (Phase 5)
# Tool 5: search_pdfs          — digital PDF library search (Phase 6)
#
# The LLM receives all 5 schemas on every /agent call and autonomously
# decides which to invoke based on query type and observation state.
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search BookSwap physical book inventory by topic, subject, or keyword. Returns re-ranked results combining semantic similarity, availability, ratings, and popularity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'beginner data structures' or 'psychological thriller'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check if specific physical books are currently available to borrow on BookSwap. Always call this after semantic_search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of book ID strings from semantic_search results"
                    }
                },
                "required": ["book_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_alternatives",
            "description": "Find similar books to a specific unavailable book. Call this when check_availability returns available=false for a book the user wants.",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "The book ID of the unavailable book to find alternatives for"
                    }
                },
                "required": ["book_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_profile",
            "description": "Retrieve a student's reading profile: borrowing history, ratings, and genre preferences. ALWAYS call this FIRST when the user asks for personalised recommendations or mentions 'based on my history / what I've read / for me'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "MongoDB ObjectId of the logged-in user (provided in the request context)"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_pdfs",
            "description": "Search the BookSwap digital PDF library — student-uploaded notes, textbook soft copies, previous year question papers, reference PDFs, leisure books in PDF form, and any other digital resource. ALWAYS call this in parallel with `semantic_search` for EVERY book-related query, regardless of whether it sounds academic, leisure, or vague. Physical books and PDFs are two parallel inventories on the BookSwap platform; calling only one of them produces a structurally incomplete answer. Skip this tool only for off-topic queries (math homework, weather, etc.) where no book would be relevant either way.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'operating systems notes' or 'DBMS previous year papers'"
                    },
                    "resource_type": {
                        "type": "string",
                        "enum": ["textbook", "notes", "previous_papers", "reference", "all"],
                        "description": "Filter by resource type. Default 'all'."
                    },
                    "department": {
                        "type": "string",
                        "description": "Optional department filter, e.g. 'CSE', 'ECE', 'Mathematics'"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# TOOL EXECUTOR
# Maps tool names from the LLM's JSON output to actual Python functions.
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_args: dict, taste_vector=None, rerank_weights=None):
    if tool_name == "semantic_search":
        from app.chapter_extractor import llm_filter_search_results

        books = list(books_collection.find({}))

        # Stage 1: cosine top-K shortlist
        candidates = search_books(tool_args["query"], books, top_k=10)

        # Stage 1b: cosine floor (cheap first-pass filter — kills out-of-domain queries)
        top_score = candidates[0].get("score", 0.0) if candidates else 0.0
        if not candidates or top_score < SEARCH_COSINE_FLOOR:
            return []

        # Stage 1c: decisive-cosine bypass.
        # When top-1 cosine is high (≥ SEARCH_COSINE_BYPASS) the user has
        # named a specific book or author — "gone girl", "silent patient",
        # "harry potter". The 5-component re-ranker is designed for browse-
        # like queries where availability and popularity legitimately break
        # ties; applied to a direct title query it can demote the actual
        # named book below an unrelated but more-borrowed title. Here we
        # honour cosine ordering directly, with availability as a tie-
        # breaker only when cosine values are effectively equal.
        if top_score >= SEARCH_COSINE_BYPASS:
            sorted_by_cosine = sorted(
                candidates,
                key=lambda b: (b.get("score", 0.0),
                               1.0 if b.get("available", True) else 0.0),
                reverse=True,
            )
            return sorted_by_cosine[:5]

        # Stage 2: LLM relevance filter — only when cosine is borderline.
        # This is where Harry-Potter-type proper-noun queries can register
        # middling cosine against unrelated titles and need a real semantic
        # check before re-ranking.
        keep_indices = llm_filter_search_results(tool_args["query"], candidates)
        if not keep_indices:
            return []
        filtered = [candidates[i] for i in keep_indices]

        # Stage 3: re-rank using 5-component formula (Phase 4).
        # taste_vector is None for anonymous users — cold-start defaults context to 0.5.
        reranked = rerank_books(filtered, taste_vector=taste_vector, weights=rerank_weights)
        return reranked[:5]

    elif tool_name == "check_availability":
        from bson import ObjectId
        results = []
        borrow_collection = db["borrowrequests"]
        for book_id in tool_args["book_ids"]:
            try:
                book = books_collection.find_one({"_id": ObjectId(book_id)})
                if book:
                    entry = {
                        "book_id": book_id,
                        "title": book.get("title", "Unknown"),
                        "available": book.get("available", True)
                    }
                    if not book.get("available", True):
                        borrow = borrow_collection.find_one(
                            {"book": ObjectId(book_id), "status": "approved"},
                            sort=[("returnDate", -1)]
                        )
                        if borrow and borrow.get("returnDate"):
                            entry["returnDate"] = borrow["returnDate"].strftime("%B %d, %Y")
                    results.append(entry)
            except Exception:
                continue
        return results

    elif tool_name == "get_alternatives":
        from bson import ObjectId
        try:
            book = books_collection.find_one({"_id": ObjectId(tool_args["book_id"])})
            if not book or "vector" not in book:
                return []
            all_books = list(books_collection.find({}))
            return get_similar_books(book["vector"], all_books, tool_args["book_id"], 3)
        except Exception:
            return []

    elif tool_name == "get_user_profile":
        # ---------------------------------------------------------------------------
        # Profile-Augmented Retrieval (Phase 5 — Chapter 3.4 of thesis)
        #
        # Queries the user's borrowing history and review ratings to build:
        #   1. genre_preferences dict  — e.g. {"FICTION": 4, "EDUCATIONAL": 2}
        #   2. taste_summary string    — human-readable, fed back to LLM
        #   3. taste_vector            — 384-dim weighted average of borrowed book
        #                               embeddings (Σ rᵢ·vᵢ / Σ rᵢ), used for
        #                               the δ·context component in re-ranking
        #   4. books_already_seen      — filter these from future recommendations
        #
        # cold-start: if user has no borrows, returns empty lists + taste_vector=None
        # ---------------------------------------------------------------------------
        from bson import ObjectId
        import numpy as np

        user_id = tool_args.get("user_id")
        if not user_id:
            return {"error": "user_id required"}

        try:
            borrow_collection = db["borrowrequests"]
            review_collection = db["reviews"]

            # Fetch all borrow records for this user
            borrows = list(borrow_collection.find({
                "borrower": ObjectId(user_id),
                "status": {"$in": ["approved", "returned"]}
            }))

            books_borrowed = []
            genre_freq = {}
            weighted_sum = None
            total_weight = 0.0
            seen_ids = []

            for borrow in borrows:
                book_id = borrow.get("book")
                if not book_id:
                    continue
                book = books_collection.find_one({"_id": book_id})
                if not book:
                    continue

                book_id_str = str(book["_id"])
                seen_ids.append(book_id_str)

                # Check if user has rated this book
                review = review_collection.find_one({
                    "author": ObjectId(user_id),
                    "book": book_id
                })
                rating = float(review["rating"]) if review and review.get("rating") else 3.5

                books_borrowed.append({
                    "title": book.get("title", "Unknown"),
                    "author": book.get("author", ""),
                    "genre": book.get("genre", ""),
                    "rating_given": rating
                })

                # Genre frequency counting
                genre = book.get("genre", "").strip()
                if genre:
                    genre_freq[genre] = genre_freq.get(genre, 0) + 1

                # Build rating-weighted sum for taste vector
                # Formula: taste_vec = Σ(rᵢ · embed(bᵢ)) / Σ rᵢ  (Hu et al. 2008)
                if "vector" in book and book["vector"]:
                    vec = np.array(book["vector"])
                    if weighted_sum is None:
                        weighted_sum = vec * rating
                    else:
                        weighted_sum += vec * rating
                    total_weight += rating

            # -----------------------------------------------------------------
            # Wishlist contribution to taste vector
            # -----------------------------------------------------------------
            # Wishlisted books are an *explicit declared interest* but without
            # a rating. We treat them as implicit positive signal at weight
            # WISHLIST_WEIGHT = 4.0 — stronger than an unrated borrow (3.5)
            # because the user actively saved them, but below a 5-star review
            # because they haven't actually consumed + enjoyed them yet.
            #
            # This is consistent with implicit-feedback literature (Hu et al.
            # 2008) where declared preferences are treated as positive signal
            # with a configurable confidence weight.
            # -----------------------------------------------------------------
            WISHLIST_WEIGHT = 4.0
            wishlist_collection = db["wishlists"]
            books_wishlisted = []
            try:
                wishlist_entries = list(wishlist_collection.find({
                    "user": ObjectId(user_id)
                }))
                for entry in wishlist_entries:
                    book_id = entry.get("book")
                    if not book_id:
                        continue
                    book = books_collection.find_one({"_id": book_id})
                    if not book:
                        continue

                    book_id_str = str(book["_id"])
                    # don't double-count a book the user both borrowed AND wishlisted
                    if book_id_str in seen_ids:
                        continue
                    seen_ids.append(book_id_str)

                    books_wishlisted.append({
                        "title":  book.get("title", "Unknown"),
                        "author": book.get("author", ""),
                        "genre":  book.get("genre", "")
                    })

                    # genre frequency boost
                    g = book.get("genre", "").strip()
                    if g:
                        genre_freq[g] = genre_freq.get(g, 0) + 1

                    # add wishlisted embedding into the weighted sum
                    if "vector" in book and book["vector"]:
                        vec = np.array(book["vector"])
                        if weighted_sum is None:
                            weighted_sum = vec * WISHLIST_WEIGHT
                        else:
                            weighted_sum += vec * WISHLIST_WEIGHT
                        total_weight += WISHLIST_WEIGHT
            except Exception:
                # wishlist collection may not exist yet in older deployments —
                # degrade silently, profile still works from borrows alone.
                pass

            # Compute final taste vector (borrows + wishlist combined)
            taste_vec = None
            if weighted_sum is not None and total_weight > 0:
                taste_vec = (weighted_sum / total_weight).tolist()

            top_genres = sorted(genre_freq, key=genre_freq.get, reverse=True)[:3]

            # -----------------------------------------------------------------
            # Cold-start: department-based taste-vector initialisation
            # (thesis §3.3.4 — "graceful cold-start handling")
            # -----------------------------------------------------------------
            # If the user has no borrow OR wishlist history, taste_vec is
            # still None at this point and the re-ranker will fall back to
            # context = 0.5 (neutral). That is graceful but uninformative.
            #
            # We can do better when the user registered with a department:
            # compute the rating-weighted centroid of books in their
            # department and use that as a starting taste vector. The user's
            # very first /agent call therefore gets recommendations leaning
            # toward their stated academic interest, even before they have
            # touched a single book.
            #
            # Set when get_user_profile is called as a tool, NOT pre-computed
            # at registration time, so a department change picked up via the
            # profile page (future feature) is reflected immediately.
            #
            # Honest framing: this is informed-prior initialisation, not real
            # personalisation. taste_summary makes the source explicit so the
            # downstream LLM can phrase recommendations as "based on your
            # department" rather than misrepresenting them as "based on your
            # reading history."
            cold_start_seeded_from_dept = False
            user_dept = ""
            if taste_vec is None:
                from bson import ObjectId as _OID  # already imported above; alias for clarity
                try:
                    user_doc = db["users"].find_one({"_id": _OID(user_id)})
                except Exception:
                    user_doc = None
                user_dept = ((user_doc or {}).get("department") or "").strip()

                if user_dept:
                    # Match books whose `department` field contains the user's
                    # department code as a substring (case-insensitive). Catches
                    # both "CSE" and "Computer Science and Engineering" stored
                    # styles without forcing the upload form to converge on one.
                    dept_books = list(books_collection.find({
                        "department": {"$regex": user_dept, "$options": "i"}
                    }))
                    dept_weighted = None
                    dept_total    = 0.0
                    n_seeded      = 0
                    for b in dept_books:
                        if "vector" not in b or not b["vector"]:
                            continue
                        r = float(b.get("avg_rating") or 3.5)
                        v = np.array(b["vector"]) * r
                        if dept_weighted is None:
                            dept_weighted = v
                        else:
                            dept_weighted += v
                        dept_total += r
                        n_seeded   += 1

                    if dept_weighted is not None and dept_total > 0:
                        taste_vec = (dept_weighted / dept_total).tolist()
                        cold_start_seeded_from_dept = True
                        # tucked into the dict via the new `cold_start_source`
                        # field below so the agent can reference it honestly.

            if books_borrowed or books_wishlisted:
                parts = []
                if books_borrowed:
                    parts.append(f"borrowed {len(books_borrowed)} book(s)")
                if books_wishlisted:
                    parts.append(f"wishlisted {len(books_wishlisted)} book(s)")
                recents = [b['title'] for b in (books_borrowed + books_wishlisted)[-3:]]
                taste_summary = (
                    f"This student has {' and '.join(parts)}. "
                    f"Preferred genres: {', '.join(top_genres) if top_genres else 'varied'}. "
                    f"Recent interests: {', '.join(recents)}."
                )
            elif cold_start_seeded_from_dept:
                taste_summary = (
                    f"New user from the {user_dept} department — no borrowing "
                    f"history yet, but taste vector seeded from the average "
                    f"of well-rated books in their department. Recommendations "
                    f"will lean academic at first; they will adjust as the "
                    f"user borrows or wishlists actual books."
                )
            else:
                taste_summary = "New user — no borrowing or wishlist history yet."

            # cold_start_source records the *origin* of taste_vec so the
            # agent can phrase recommendations honestly:
            #   "history"    — from real borrows / wishlist (always preferred)
            #   "department" — informed prior from the user's stated department
            #   "none"       — taste_vec is None; re-ranker will fall back to 0.5
            if books_borrowed or books_wishlisted:
                cold_start_source = "history"
            elif cold_start_seeded_from_dept:
                cold_start_source = "department"
            else:
                cold_start_source = "none"

            return {
                "books_borrowed":    books_borrowed,
                "books_wishlisted":  books_wishlisted,
                "genre_preferences": genre_freq,
                "top_genres":        top_genres,
                "taste_summary":     taste_summary,
                "taste_vector":      taste_vec,    # returned to agent for context
                "books_already_seen": seen_ids,    # agent should exclude these
                "cold_start_source": cold_start_source,
                "user_department":   user_dept,    # "" when not registered with one
            }

        except Exception as e:
            return {"error": f"Could not load profile: {str(e)}"}

    elif tool_name == "search_pdfs":
        # ---------------------------------------------------------------------------
        # PDF Library Search (Phase 6 — System 2: Digital Resource Library)
        #
        # Searches the `pdfs` collection using vector similarity.
        # Falls back to MongoDB text search if no vectors exist yet.
        # Filters by resource_type and department if provided.
        # ---------------------------------------------------------------------------
        pdfs_collection = db["pdfs"]
        query = tool_args.get("query", "")
        resource_type = tool_args.get("resource_type", "all")
        department = tool_args.get("department")

        # Build filter
        mongo_filter = {}
        if resource_type and resource_type != "all":
            mongo_filter["resource_type"] = resource_type
        if department:
            mongo_filter["department"] = {"$regex": department, "$options": "i"}

        all_pdfs = list(pdfs_collection.find(mongo_filter))

        if not all_pdfs:
            return []

        # Vector search if embeddings exist
        pdfs_with_vectors = [p for p in all_pdfs if p.get("embedding")]
        if pdfs_with_vectors:
            from app.embeddings import generate_embedding
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            query_vec = np.array([generate_embedding(query)])
            results = []
            for pdf in pdfs_with_vectors:
                pdf_vec = np.array([pdf["embedding"]])
                score = float(cosine_similarity(query_vec, pdf_vec)[0][0])
                results.append({
                    "_id": str(pdf["_id"]),
                    "title": pdf.get("title", "Unknown"),
                    "subject": pdf.get("subject", ""),
                    "course": pdf.get("course", ""),
                    "professor": pdf.get("professor", ""),
                    "department": pdf.get("department", ""),
                    "resource_type": pdf.get("resource_type", ""),
                    "cloudinary_url": pdf.get("cloudinary_url", ""),
                    "description": pdf.get("description", ""),
                    "download_count": pdf.get("download_count", 0),
                    "score": round(score, 4)
                })
            results.sort(key=lambda x: x["score"], reverse=True)
            return [r for r in results if r["score"] > 0.1][:5]

        # Fallback: return first 5 matching by filter (no vector yet)
        results = []
        for pdf in all_pdfs[:5]:
            results.append({
                "_id": str(pdf["_id"]),
                "title": pdf.get("title", "Unknown"),
                "subject": pdf.get("subject", ""),
                "resource_type": pdf.get("resource_type", ""),
                "cloudinary_url": pdf.get("cloudinary_url", ""),
                "description": pdf.get("description", ""),
            })
        return results

    return {"error": f"Unknown tool: {tool_name}"}

# ---------------------------------------------------------------------------
# SESSION MEMORY (Phase 7 — Chapter 3.5 of thesis)
#
# Two layers of memory:
#
#   Session memory:    full turn history for current session, stored in MongoDB
#                      `conversations` collection. Loaded at session start and
#                      appended on every turn. Injected into Groq messages list
#                      so the LLM sees the complete current conversation.
#
#   Persistent memory: context_summary from the user's most recent PAST session
#                      is injected as a system note at the start of a new session.
#                      Gives continuity across logins without a full history reload.
#
# Context compression: after 10 turns, summarise with Groq and store as
#                      context_summary. Prevents token overflow for long chats.
#
# Distinguished from persistent user profile (Phase 5):
#   - Profile = static behavioural fingerprint (borrow history, taste vector)
#   - Session memory = dynamic conversational context (what was said this chat)
# ---------------------------------------------------------------------------

def load_session(session_id: str, user_id: Optional[str] = None) -> dict:
    """Get or create a conversation session document."""
    now = datetime.now(timezone.utc)
    session = conversations_collection.find_one({"session_id": session_id})
    if not session:
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": now,
            "last_active": now,
            "turns": [],
            "context_summary": None
        }
        conversations_collection.insert_one(session)
    return session


def save_turn(session_id: str, role: str, content: str, tool_call_id: Optional[str] = None):
    """Append a single turn to the session's turns array and update last_active."""
    turn = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc)
    }
    if tool_call_id:
        turn["tool_call_id"] = tool_call_id
    conversations_collection.update_one(
        {"session_id": session_id},
        {
            "$push": {"turns": turn},
            "$set": {"last_active": datetime.now(timezone.utc)}
        }
    )


def compress_context(session_id: str):
    """
    After every 10 turns, summarise the conversation and store as context_summary.
    This keeps the token count manageable for long sessions.
    Academic framing: token-budget-aware memory management.
    """
    session = conversations_collection.find_one({"session_id": session_id})
    if not session or len(session.get("turns", [])) < 10:
        return

    turns_text = "\n".join(
        f"{t['role'].upper()}: {t['content']}"
        for t in session["turns"][-10:]
        if t.get("content")
    )

    try:
        summary_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    "Summarise this BookSwap AI conversation in exactly 3 sentences. "
                    "Include: what the student was looking for, what was recommended, "
                    "and any preference or constraint they mentioned.\n\n"
                    f"{turns_text}"
                )
            }],
            temperature=0,
            max_tokens=150
        )
        summary = summary_response.choices[0].message.content or ""
        conversations_collection.update_one(
            {"session_id": session_id},
            {"$set": {"context_summary": summary, "turns": []}}
        )
    except Exception:
        pass  # compression is best-effort; don't crash the agent


def get_session_messages(session_id: str, user_id: Optional[str] = None) -> list:
    """
    Build the messages list to inject into Groq.
    Returns: list of {role, content} dicts representing the conversation history.

    Priority:
      1. If existing session has turns → return them as messages
      2. If no turns but context_summary exists → inject as a brief system note
      3. If new user, look for most recent past session → inject its summary
      4. Otherwise → empty list (fresh start)
    """
    session = load_session(session_id, user_id)
    turns = session.get("turns", [])

    if turns:
        # Convert stored turns to Groq message format
        messages = []
        for t in turns:
            msg = {"role": t["role"], "content": t.get("content", "")}
            if t.get("tool_call_id"):
                msg["tool_call_id"] = t["tool_call_id"]
            messages.append(msg)
        return messages

    # No turns in current session — check for summary from this session
    if session.get("context_summary"):
        return [{
            "role": "system",
            "content": f"[Conversation summary so far]: {session['context_summary']}"
        }]

    # New session — look for a past session summary for this user (persistent memory)
    if user_id:
        past = conversations_collection.find_one(
            {
                "user_id": user_id,
                "session_id": {"$ne": session_id},
                "context_summary": {"$ne": None}
            },
            sort=[("last_active", -1)]
        )
        if past and past.get("context_summary"):
            return [{
                "role": "system",
                "content": (
                    "[Note: This student has used BookSwap AI before. "
                    f"Previous session summary]: {past['context_summary']}"
                )
            }]

    return []


# ---------------------------------------------------------------------------
# REACT LOOP — extracted as a standalone function.
#
# Why extracted? The /evaluate endpoint (Phase 11) calls this function
# directly for 35 test queries without going through HTTP. If the loop
# lived inside the route handler, the evaluation would need to make HTTP
# calls to itself — circular and fragile.
#
# Returns: (response_text, tools_called, iteration_count)
# ---------------------------------------------------------------------------

def _extract_cited(tool_name: str, result) -> list[dict]:
    """
    Pull {id, title, type} triples out of a tool result so the chat
    widget can hyperlink any of those titles when they appear in the
    agent's free-text response.

    Best-effort — only handles tool results whose items are dicts with
    a recognisable id+title shape.  Tool results that don't fit
    (get_user_profile's taste-vector payload, etc.) just return [].
    """
    cited: list[dict] = []
    if not isinstance(result, list):
        return cited
    for item in result:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if not title:
            continue
        if tool_name == "search_pdfs" or "pdf_id" in item or "cloudinary_url" in item:
            obj_id = item.get("pdf_id") or item.get("_id")
            kind   = "pdf"
        else:
            obj_id = item.get("book_id") or item.get("_id")
            kind   = "book"
        if obj_id and title:
            cited.append({
                "id":    str(obj_id),
                "title": str(title),
                "type":  kind,
            })
    return cited


def run_react_loop(
    messages: list,
    available_tools: list,
    max_iterations: int = 10,
    rerank_weights: dict = None
) -> tuple[str, List[str], int, list[dict]]:
    """
    Core ReAct agent loop (Yao et al. 2022).

    State tracked across iterations:
      taste_vector    — set when get_user_profile is called; threaded
                        into subsequent semantic_search calls for
                        personalised re-ranking.
      cited_resources — every {id, title, type} triple seen in any
                        tool result during this turn, deduplicated by
                        id, returned to the caller so the frontend can
                        linkify those titles in the final response.

    Returns: (response_text, tools_called, iterations, cited_resources)
    """
    tools_called: list[str]    = []
    cited_resources: list[dict] = []
    taste_vector = None   # updated when get_user_profile runs

    def _add_cited(items: list[dict]):
        seen_ids = {c["id"] for c in cited_resources}
        for c in items:
            if c["id"] not in seen_ids:
                cited_resources.append(c)
                seen_ids.add(c["id"])

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")

        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
                temperature=0       # deterministic — identical input → identical output
            )
        except Exception as e:
            error_msg = str(e)
            print(f"GROQ ERROR: {error_msg}")
            if "rate_limit" in error_msg or "429" in error_msg:
                return "I'm a bit busy right now, please try again in a few minutes!", tools_called, iteration, cited_resources
            continue

        msg = response.choices[0].message

        # No tool calls means LLM has enough information to answer
        if not msg.tool_calls:
            preview = (msg.content or '')[:200].encode('ascii', errors='replace').decode('ascii')
            print(f"FINAL: {preview}")
            return msg.content or "I couldn't find an answer. Please try again.", tools_called, iteration, cited_resources

        # Append assistant message (with its tool_calls) to conversation history
        messages.append(msg)

        # Execute each tool the LLM requested and feed results back
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"TOOL: {tool_name}({tool_args})")
            tools_called.append(tool_name)

            # Pass taste_vector + rerank_weights into semantic_search
            # so re-ranking can use the user's profile if available
            result = execute_tool(
                tool_name, tool_args,
                taste_vector=taste_vector,
                rerank_weights=rerank_weights
            )

            # If get_user_profile returned a taste_vector, capture it for
            # subsequent semantic_search calls in this same session
            if tool_name == "get_user_profile" and isinstance(result, dict):
                taste_vector = result.get("taste_vector")  # may be None (cold start)

            # Capture any books / PDFs surfaced by this tool call so the
            # chat widget can render them as clickable links.
            _add_cited(_extract_cited(tool_name, result))

            print(f"RESULT: {json.dumps(result, default=str)[:300]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str)
            })

    # Safety fallback — only fires when the agent loop genuinely cannot
    # converge on a final answer within max_iterations. The pilot user-
    # test session triggered this once on a "ref book for fluid machines
    # course" query, and without the diagnostic line below we could not
    # tell whether the agent had been stuck looping on one tool, calling
    # tools sequentially that should have been parallel, or getting
    # confused by an empty inventory result. Now we print the full tool-
    # call trail and the last few message roles so future debugging on
    # Render logs is one log-grep away.
    last_roles = [m.get("role") for m in messages[-6:]]
    print(
        f"AGENT MAX-ITERATIONS HIT — tools_called={tools_called!r} "
        f"last_message_roles={last_roles!r} max_iterations={max_iterations}",
        flush=True,
    )
    # Friendlier safety message — the previous "please try rephrasing"
    # was unhelpful for the study-help failure mode (user asks "what
    # should I study for tomorrow's wastewater engineering quiz" and
    # the agent loops because no wastewater engineering material is
    # in inventory). Point them at the curriculum upload, which is
    # the actual path that solves their problem.
    return (
        "I couldn't find enough material in BookSwap's library to fully "
        "answer that. If you're studying for a course, try uploading your "
        "lecture plan PDF on the Curriculum page (/curriculum) — the "
        "system maps each unit to specific chapters in the professor's "
        "recommended textbooks. Otherwise please rephrase your question.",
        tools_called,
        max_iterations,
        cited_resources,
    )

# ---------------------------------------------------------------------------
# AGENT ENDPOINT
# ---------------------------------------------------------------------------

@router.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    # Assign or preserve session_id.
    # New session: frontend generates a UUID and stores it in sessionStorage.
    # Subsequent turns: frontend sends the same session_id back → history loaded.
    session_id = request.session_id or str(uuid.uuid4())
    user_id = request.user_id  # None for unauthenticated / guests

    # Pick the agent system prompt by version. None → latest (AGENT_LATEST).
    # The request path uses the registry instead of the legacy SYSTEM_PROMPT
    # constant so evaluation runs can swap versions without touching code.
    from app.prompts import get_agent_prompt, AGENT_LATEST
    from app.experiment_log import log_agent_run

    active_version = request.prompt_version or AGENT_LATEST
    try:
        active_prompt = get_agent_prompt(active_version)
    except ValueError:
        # Unknown version → fall back to latest, but log the anomaly
        active_version = AGENT_LATEST
        active_prompt  = get_agent_prompt(active_version)

    # --- Build messages list ---
    # Structure: system_prompt → [session history] → current user message
    #
    # session history: full turn array for current session, OR
    #                  context_summary from past session (persistent memory)
    # This gives the LLM awareness of what was said earlier in the conversation.
    session_history = get_session_messages(session_id, user_id)

    messages = [{"role": "system", "content": active_prompt}]
    messages.extend(session_history)
    messages.append({"role": "user", "content": request.message})

    # Save user turn to session store
    save_turn(session_id, "user", request.message)

    # Run the ReAct loop
    loop_error = None
    try:
        response_text, tools_called, iterations, cited_resources = run_react_loop(messages, tools)
    except Exception as e:
        loop_error = str(e)
        response_text, tools_called, iterations, cited_resources = (
            "Sorry, something went wrong. Please try again.", [], 0, []
        )

    # Save assistant response to session store
    save_turn(session_id, "assistant", response_text)

    # Compress context if session is getting long (token budget management)
    session = conversations_collection.find_one({"session_id": session_id})
    if session and len(session.get("turns", [])) >= 10:
        compress_context(session_id)

    # Thesis Chapter 5: append one row per response to agent_runs.jsonl —
    # lets us compute Tool Precision, Hallucination Rate, etc. offline by
    # reading the log instead of re-running the agent.
    log_agent_run(
        prompt_version=active_version,
        user_id=user_id,
        session_id=session_id,
        query=request.message,
        response=response_text,
        tools_called=tools_called,
        iterations=iterations,
        error=loop_error,
    )

    return AgentResponse(
        response=response_text,
        session_id=session_id,
        tools_called=tools_called,
        iterations=iterations,
        cited_resources=cited_resources,
    )

# ---------------------------------------------------------------------------
# EXISTING ENDPOINTS — unchanged
# ---------------------------------------------------------------------------

@router.post("/embed-all")
def embed_all(force: bool = False):
    """
    Backfill vector embeddings for every book in the inventory.

    Idempotent: skips books that already have a non-empty `vector` field so
    it's safe to re-run on every deploy. Pass ?force=true to re-embed
    everything (use when the embedding model changes).

    This is the fix for "all books show 'Not in library'" — seeded books
    have no vector, so `books_with_vectors` filter in chapter_extractor
    returns an empty list → every match falls through.
    """
    embedded = 0
    skipped  = 0
    for book in books_collection.find({}):
        if not force and book.get("vector"):
            skipped += 1
            continue
        text = f"{book.get('title', '')} {book.get('author', '')} {book.get('genre', '')} {book.get('description', '')}"
        vector = generate_embedding(text)
        books_collection.update_one(
            {"_id": book["_id"]},
            {"$set": {"vector": vector}}
        )
        embedded += 1
    return {"embedded": embedded, "skipped_already_embedded": skipped}

@router.post("/search")
def search(request: SearchRequest):
    """
    Search-bar endpoint hit by Node.js GET /books?query=...

    Architectural choice (thesis §3.3 — revised after user testing
    surfaced "inspirational books" returning Verity / Project Hail Mary
    / a probability textbook instead of Atomic Habits):

    Two-stage retrieval at the search bar, with a high-cosine bypass
    for direct queries.

      Stage 1  — cosine top-K shortlist + floor (rejects out-of-domain
                  queries like "Harry Potter" against an OS-only inventory)
      Stage 1c — decisive-cosine bypass: when the top match has cosine
                  >= SEARCH_COSINE_BYPASS, the user has named a target
                  ("gone girl" -> Gone Girl) and the LLM filter would
                  only second-guess it. Skip Stage 2.
      Stage 2  — LLM relevance filter for borderline cosine. This is
                  where vague topic queries ("inspirational books",
                  "thrillers", "easy read") need a semantic check
                  beyond word-level cosine, because cosine on a 384-dim
                  Sentence-BERT embedding is sensitive to vocabulary
                  drift (e.g. "inspirational" embedding does not strongly
                  correlate with "habits productivity behaviour change",
                  even though Atomic Habits is the canonical inspirational
                  book in our inventory).

    No re-ranking on this surface — that lives on the chat-agent's
    semantic_search tool only. See thesis §3.3 for the retrieval-vs-
    recommendation split (Burke, 2002).
    """
    from app.chapter_extractor import llm_filter_search_results

    books = list(books_collection.find({}))
    candidates = search_books(request.query, books, top_k=10)

    # Out-of-distribution guard. One cheap float comparison.
    top_score = candidates[0].get("score", 0.0) if candidates else 0.0
    if not candidates or top_score < SEARCH_COSINE_FLOOR:
        return {"results": []}

    # Decisive-cosine bypass: if cosine alone is conclusive, skip the
    # LLM call (saves ~400 ms latency on direct title/author queries).
    if top_score >= SEARCH_COSINE_BYPASS:
        return {"results": candidates[: request.top_k]}

    # LLM relevance filter. Returns the indices of candidates that the
    # LLM judges genuinely relevant; if the LLM rejects all, return
    # empty so the UI shows "no results" honestly rather than noise.
    keep_indices = llm_filter_search_results(request.query, candidates)
    if not keep_indices:
        return {"results": []}
    filtered = [candidates[i] for i in keep_indices]
    return {"results": filtered[: request.top_k]}

@router.post("/embed-book")
def embed_book(request: EmbedRequest):
    from bson import ObjectId
    book = books_collection.find_one({"_id": ObjectId(request.book_id)})
    if not book:
        return {"error": "Book not found"}
    text = f"{book.get('title', '')} {book.get('author', '')} {book.get('genre', '')} {book.get('description', '')}"
    vector = generate_embedding(text)
    books_collection.update_one(
        {"_id": ObjectId(request.book_id)},
        {"$set": {"vector": vector}}
    )
    return {"message": "Embedding stored successfully"}

@router.post("/similar-books")
def similar_books(request: SimilarBooksRequest):
    from bson import ObjectId
    book = books_collection.find_one({"_id": ObjectId(request.book_id)})
    if not book:
        return {"error": "Book not found"}
    if "vector" not in book:
        return {"error": "Book does not have an embedding"}
    org_vector = book["vector"]
    books = list(books_collection.find({}))
    results = get_similar_books(org_vector, books, request.book_id, request.top_k)
    return {"results": results}

@router.post("/recommend-personal")
def recommend_personal(request: PersonalRecommendationRequest):
    from bson import ObjectId
    weighted_books = []
    library_ids = []
    for item in request.books:
        book = books_collection.find_one({"_id": ObjectId(item.book_id)})
        if not book or "vector" not in book:
            continue
        weighted_books.append({"vector": book["vector"], "rating": item.rating})
        library_ids.append(item.book_id)
    if not weighted_books:
        return {"results": []}
    all_books = list(books_collection.find({}))
    results = get_personal_recommendations(weighted_books, all_books, library_ids, request.top_k)
    return {"results": results}

@router.post("/check-availability")
def check_availability(request: AvailabilityRequest):
    from bson import ObjectId
    results = []
    for book_id in request.book_ids:
        book = books_collection.find_one({"_id": ObjectId(book_id)})
        if book:
            results.append({
                "book_id": book_id,
                "title": book.get("title", "Unknown"),
                "available": book.get("available", True)
            })
    return {"results": results}

# ---------------------------------------------------------------------------
# EMBED-PDF ENDPOINT
# Called by Node.js after a PDF is uploaded to Cloudinary.
# Downloads the PDF, extracts chapter headings, generates an embedding
# for the title+subject text, and stores both in MongoDB pdfs collection.
# ---------------------------------------------------------------------------

class EmbedPdfRequest(BaseModel):
    pdf_id: str   # MongoDB ObjectId string of the Pdf document

@router.post("/embed-pdf")
def embed_pdf_resource(request: EmbedPdfRequest):
    """
    POST /embed-pdf
    Body: { "pdf_id": "<MongoDB ObjectId string>" }

    Downloads the PDF from Cloudinary, extracts chapter headings using
    PyMuPDF, generates a 384-dim embedding for the title+subject text,
    and saves both to the pdfs collection.
    """
    import requests as http_requests
    from bson import ObjectId
    from app.chapter_extractor import extract_chapter_headings_from_bytes

    pdfs_collection = db["pdfs"]

    try:
        pdf_doc = pdfs_collection.find_one({"_id": ObjectId(request.pdf_id)})
    except Exception:
        return {"error": "Invalid pdf_id"}

    if not pdf_doc:
        return {"error": "PDF document not found"}

    # Generate embedding from title + subject
    text_for_embedding = (
        f"{pdf_doc.get('title', '')} "
        f"{pdf_doc.get('subject', '')} "
        f"{pdf_doc.get('course', '')} "
        f"{pdf_doc.get('description', '')}"
    ).strip()
    embedding = generate_embedding(text_for_embedding)

    # Try to extract chapter headings
    chapter_headings = []
    cloudinary_url = pdf_doc.get("cloudinary_url", "")
    if cloudinary_url:
        try:
            resp = http_requests.get(cloudinary_url, timeout=15)
            if resp.status_code == 200:
                chapter_headings = extract_chapter_headings_from_bytes(resp.content)
        except Exception:
            pass  # headings are best-effort

    pdfs_collection.update_one(
        {"_id": ObjectId(request.pdf_id)},
        {"$set": {
            "embedding": embedding,
            "chapter_headings": chapter_headings
        }}
    )

    return {
        "message": "PDF embedded successfully",
        "chapters_found": len(chapter_headings)
    }


@router.post("/embed-all-pdfs")
def embed_all_pdfs(force: bool = False):
    """
    Backfill `embedding` + `chapter_headings` for every PDF in the pdfs collection.

    Idempotent: skips PDFs that already have an `embedding` field. Pass
    ?force=true to re-embed everything (use when embedding model changes
    or chapter-extraction heuristics improve).

    Why this exists: PDFs uploaded before the /embed-pdf hook was wired,
    or uploads where the hook failed silently, have no `embedding` →
    `pdfs_with_vectors` filter in chapter_extractor returns empty →
    professor-book matching can't hit a student-uploaded Galvin PDF →
    unit→chapter mapping can't use its TOC. One call to this endpoint
    fixes all downstream gaps.
    """
    import requests as http_requests
    from app.chapter_extractor import extract_chapter_headings_from_bytes

    pdfs_collection = db["pdfs"]
    embedded = 0
    skipped  = 0
    failed   = 0

    for pdf_doc in pdfs_collection.find({}):
        if not force and pdf_doc.get("embedding"):
            skipped += 1
            continue

        text_for_embedding = (
            f"{pdf_doc.get('title', '')} "
            f"{pdf_doc.get('subject', '')} "
            f"{pdf_doc.get('course', '')} "
            f"{pdf_doc.get('description', '')}"
        ).strip()
        try:
            embedding = generate_embedding(text_for_embedding)
        except Exception:
            failed += 1
            continue

        chapter_headings = []
        cloudinary_url = pdf_doc.get("cloudinary_url", "")
        if cloudinary_url:
            try:
                resp = http_requests.get(cloudinary_url, timeout=15)
                if resp.status_code == 200:
                    chapter_headings = extract_chapter_headings_from_bytes(resp.content)
            except Exception:
                pass   # headings are best-effort, embedding still counts

        pdfs_collection.update_one(
            {"_id": pdf_doc["_id"]},
            {"$set": {
                "embedding": embedding,
                "chapter_headings": chapter_headings,
            }}
        )
        embedded += 1

    return {
        "embedded": embedded,
        "skipped_already_embedded": skipped,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# /rescan-pdf-chapters — re-run chapter extraction on a single existing PDF.
#
# Why this exists: the default `extract_chapter_headings_from_bytes()` run
# at upload time stores ONLY Tier 2a (embedded TOC) entries — chapter-level
# granularity. For curriculum matching we sometimes want Tier 2b (font-scan)
# subsection headings as well, which are off by default (experimental, slower).
#
# This endpoint downloads the PDF from Cloudinary again and re-extracts with
# the include_page_scan flag, so we don't have to re-upload the file.
# Used for thesis §3.7 Tier 2b ablation: compare curriculum match accuracy
# with vs without font-scan enabled.
# ---------------------------------------------------------------------------
class RescanRequest(BaseModel):
    pdf_id: str
    include_page_scan: bool = False


@router.post("/rescan-pdf-chapters")
def rescan_pdf_chapters(request: RescanRequest):
    """
    Re-run chapter_headings extraction on a single PDF.

    Body: { "pdf_id": "<ObjectId>", "include_page_scan": true }
    Returns: number of headings found + first 20 as a sample so we can eyeball
    them in Swagger before trusting the update.
    """
    import requests as http_requests
    from bson import ObjectId
    from app.chapter_extractor import extract_chapter_headings_from_bytes

    pdfs_collection = db["pdfs"]
    try:
        pdf_doc = pdfs_collection.find_one({"_id": ObjectId(request.pdf_id)})
    except Exception:
        return {"error": "Invalid pdf_id"}
    if not pdf_doc:
        return {"error": "PDF document not found"}

    cloudinary_url = pdf_doc.get("cloudinary_url", "")
    if not cloudinary_url:
        return {"error": "No cloudinary_url on PDF document"}

    try:
        resp = http_requests.get(cloudinary_url, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Could not download PDF: {e}"}

    headings = extract_chapter_headings_from_bytes(
        resp.content,
        include_page_scan=request.include_page_scan,
    )

    pdfs_collection.update_one(
        {"_id": ObjectId(request.pdf_id)},
        {"$set": {"chapter_headings": headings}}
    )

    return {
        "ok": True,
        "pdf_id": request.pdf_id,
        "title": pdf_doc.get("title", ""),
        "chapters_found": len(headings),
        "include_page_scan": request.include_page_scan,
        "sample": [
            {"title": h.get("title"), "page": h.get("page")}
            for h in headings[:20]
        ],
    }


@router.get("/debug-vectors")
def debug_vectors():
    results = []
    for book in books_collection.find({}):
        vec = book.get("vector", None)
        results.append({
            "title": book.get("title", "Unknown"),
            "has_vector": vec is not None,
            "vector_length": len(vec) if vec else 0,
            "first_3_values": vec[:3] if vec else []
        })
    return {"books": results}


# ---------------------------------------------------------------------------
# CURRICULUM ENDPOINT — Phase 10 (Chapter 3.7 of thesis)
#
# Accepts a lecture plan PDF (multipart upload), runs the full
# chapter_extractor pipeline, and returns topic-to-book matches.
# The Node.js frontend POSTs the PDF as multipart/form-data.
# ---------------------------------------------------------------------------

from fastapi import UploadFile, File, Form

@router.post("/curriculum")
async def curriculum_match(
    file: UploadFile = File(...),
    prompt_version: Optional[str] = Form(None),
):
    """
    POST /curriculum
    Upload a lecture plan PDF → structured curriculum match.

    Response:
      {
        "course_name": str,
        "course_code": str,
        "department": str,
        "recommended_books": [
          {
            "recommended_title": str,   <- what professor wrote
            "found": bool,              <- is it in our library?
            "book_title": str,
            "book_author": str,
            "book_id": str,
            "available": bool,          <- available to borrow right now?
            "match_score": float
          }
        ],
        "unit_matches": [
          {
            "unit_no": int,
            "unit_title": str,
            "book_title": str,
            "book_type": "physical",
            "source": "professor_recommended" | "ai_suggested",
            "available": bool,
            "suggested_chapter": str,
            "chapter_page": int | null,  <- page number if PDF TOC available
            "match_score": float
          }
        ],
        "error": str | null
      }
    """
    from app.chapter_extractor import process_curriculum_pdf
    pdf_bytes = await file.read()
    # Pass physical books AND digital PDFs
    # Physical books: borrowable copies in the books collection
    # Digital PDFs: uploaded textbooks in the pdfs collection — these have
    #               get_toc() page numbers from /embed-pdf, giving accurate chapters
    books = list(books_collection.find({}))
    pdfs  = list(db["pdfs"].find({}))
    result = process_curriculum_pdf(
        pdf_bytes,
        books,
        pdfs,
        prompt_version=prompt_version,   # None → falls back to latest
        pdf_filename=file.filename,      # logged for thesis reproducibility
    )
    return result


# ---------------------------------------------------------------------------
# /compress-pdf — lossless PDF compression for the upload pipeline.
#
# Why this exists:
#   Cloudinary's free tier rejects raw uploads larger than 10 MB. Real
#   textbooks scan to 20-50 MB, which the user can't be expected to
#   compress by hand before every upload. The Node side runs every PDF
#   upload through this endpoint when its size is above ~9.5 MB; we
#   apply PyMuPDF's lossless deflate options (garbage collection +
#   stream + image + font deflate + clean) and return the compressed
#   bytes so Node can push them on to Cloudinary.
#
# Lossy compression (rendering pages as JPEGs) is deliberately NOT
# attempted here even when lossless isn't enough. Reason: the curriculum
# matcher in §3.7 depends on extractable text and embedded TOCs;
# replacing pages with rasterised images would silently break that
# feature for every aggressively-compressed upload. When lossless
# isn't enough, we return 413 with a clear message asking the user
# to compress externally — that is honest and preserves downstream
# functionality.
# ---------------------------------------------------------------------------
from fastapi.responses import Response, JSONResponse

@router.post("/compress-pdf")
async def compress_pdf(file: UploadFile = File(...)):
    import io
    pdf_bytes = await file.read()
    original_size = len(pdf_bytes)
    target_bytes = int(9.5 * 1024 * 1024)   # under Cloudinary's 10 MB cap

    # Already small enough — return as-is, no work to do.
    if original_size <= target_bytes:
        return Response(content=pdf_bytes, media_type="application/pdf")

    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        out = io.BytesIO()
        doc.save(
            out,
            garbage=4,           # full xref garbage collection
            deflate=True,        # compress streams
            deflate_images=True, # compress embedded images
            deflate_fonts=True,  # compress embedded fonts
            clean=True,          # remove redundant content
        )
        doc.close()
        compressed = out.getvalue()
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Compression failed: {type(e).__name__}: {e}"},
        )

    if len(compressed) <= target_bytes:
        return Response(content=compressed, media_type="application/pdf")

    # Lossless wasn't enough. Refuse rather than silently degrade.
    return JSONResponse(
        status_code=413,
        content={
            "error": "PDF still exceeds 10 MB after lossless compression",
            "original_mb":   round(original_size / 1024 / 1024, 1),
            "compressed_mb": round(len(compressed)  / 1024 / 1024, 1),
            "limit_mb":      10,
            "suggestion": (
                "Use a desktop tool such as iLovePDF, Smallpdf, or Adobe "
                "Acrobat's compression preset to reduce the file further "
                "before uploading. Aggressive in-pipeline compression would "
                "render the PDF as images and break the curriculum-matcher's "
                "text-extraction step."
            ),
        },
    )


# ---------------------------------------------------------------------------
# EVALUATION ENDPOINT — Phase 11 (Chapter 5 of thesis)
#
# Runs 35 pre-defined test queries through the agent (or a reduced set for
# specific experiments) and computes 5 metrics:
#
#   1. Tool Precision      — fraction of tool calls that are appropriate
#   2. Hallucination Rate  — fraction of responses mentioning titles NOT in tool results
#   3. Response Relevance  — keyword match score between query and response
#   4. Task Completion Rate— fraction where agent produced a non-error response
#   5. Prompt Adherence    — fraction where forbidden patterns (JSON, IDs) absent
#
# Plus the novel Curriculum Coverage Score (CCS) for curriculum queries:
#   CCS = |matched_topics| / |total_topics_in_plan|
#
# prompt_version: "v1"/"v2"/"v3"/"v4" — swaps SYSTEM_PROMPT before running
# This is the core of RQ2: "Which prompt engineering technique combination
# produces the best agent behaviour?"
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    prompt_version: Optional[str] = "v4"   # which prompt to test
    query_subset: Optional[List[str]] = None  # override with custom queries

@router.post("/evaluate")
def evaluate(request: EvalRequest):
    from app.prompts import get_agent_prompt, AGENT_LATEST
    from app.experiment_log import log_agent_run

    # Select prompt version from the registry — same source of truth as /agent.
    active_version = request.prompt_version or AGENT_LATEST
    try:
        active_prompt = get_agent_prompt(active_version)
    except ValueError:
        active_version = AGENT_LATEST
        active_prompt  = get_agent_prompt(active_version)

    # 35 test queries covering all agent capabilities
    # (5 categories × 7 queries each)
    test_queries = request.query_subset or [
        # Category 1: Direct book search (7 queries)
        "Find me thriller books",
        "I need books on data structures",
        "Do you have any self-help books?",
        "Show me books by Colleen Hoover",
        "Find books about leadership",
        "I want to read a mystery novel",
        "Any motivational books available?",
        # Category 2: Availability + alternatives (7 queries)
        "Is Verity available to borrow?",
        "Can I borrow The Housemaid?",
        "Find me something like The Silent Patient",
        "What thriller books can I borrow right now?",
        "Show me available psychology books",
        "I want to borrow a fiction book this week",
        "Any books available on leadership?",
        # Category 3: Vague / clarification needed (7 queries)
        "Suggest me a good book",
        "What are some easy reads?",
        "I need something interesting",
        "Recommend me something",
        "Any new arrivals?",
        "What's popular right now?",
        "I'm bored, help me find a book",
        # Category 4: Off-topic / boundary tests (7 queries)
        "What is the weather today?",
        "Help me write a Python script",
        "Who is the Prime Minister of India?",
        "Can you book me a flight?",
        "What is machine learning?",
        "Tell me a joke",
        "Help me with my math homework",
        # Category 5: Digital PDF / study material (7 queries)
        "Find OS notes for CSE",
        "Any DBMS previous year papers?",
        "Show me reference material for algorithms",
        "I need study material for networks",
        "Find textbooks for discrete mathematics",
        "Any CSE department notes?",
        "Previous year papers for data structures",
    ]

    results = []
    hallucination_count = 0
    task_complete_count = 0
    adherence_count = 0
    total_tool_calls = 0
    appropriate_tool_calls = 0
    total_relevance = 0.0

    FORBIDDEN_PATTERNS = [
        '"_id"', '"book_id"', '"tool_call"', 'function_name',
        'ObjectId', '"score":', '"vector"'
    ]

    for query in test_queries:
        messages = [
            {"role": "system", "content": active_prompt},
            {"role": "user", "content": query}
        ]

        response_text, tools_called, iterations, _cited = run_react_loop(
            messages, tools, max_iterations=5
        )

        # Log this individual query run so the Chapter 5 ablation table can be
        # built by reading experiments/agent_runs.jsonl, filtered on
        # prompt_version. Each /evaluate invocation produces 35 rows here.
        log_agent_run(
            prompt_version=active_version,
            user_id=None,
            session_id=f"eval-{active_version}",
            query=query,
            response=response_text,
            tools_called=tools_called,
            iterations=iterations,
            error=None,
        )

        # ----- Metric 1: Task Completion -----
        completed = (
            response_text
            and not response_text.startswith("Sorry")
            and len(response_text) > 20
        )
        if completed:
            task_complete_count += 1

        # ----- Metric 2: Hallucination Rate -----
        # A response hallucinated if it mentions a title not in any tool result
        # Approximation: check if response contains titles NOT in our DB
        db_titles = [b.get("title", "").lower() for b in books_collection.find({}, {"title": 1})]
        words = response_text.lower()
        # If agent called no search tool but still recommends books → hallucination
        search_tools = [t for t in tools_called if "search" in t]
        has_book_mention = any(title in words for title in db_titles if title)
        if has_book_mention and not search_tools:
            hallucination_count += 1

        # ----- Metric 3: Response Relevance -----
        query_keywords = set(query.lower().split())
        response_keywords = set(response_text.lower().split())
        overlap = len(query_keywords & response_keywords)
        relevance = overlap / max(len(query_keywords), 1)
        total_relevance += relevance

        # ----- Metric 4: Tool Precision -----
        # Off-topic queries should call 0 tools; book queries should call search tools
        is_book_query = any(kw in query.lower() for kw in [
            "book", "borrow", "find", "suggest", "recommend", "pdf", "notes", "thriller",
            "available", "fiction", "textbook", "previous year", "papers"
        ])
        for tc in tools_called:
            total_tool_calls += 1
            if is_book_query and tc in ["semantic_search", "check_availability", "get_alternatives", "search_pdfs"]:
                appropriate_tool_calls += 1
            elif not is_book_query and len(tools_called) == 0:
                appropriate_tool_calls += 1

        # ----- Metric 5: Prompt Adherence -----
        has_forbidden = any(fp in response_text for fp in FORBIDDEN_PATTERNS)
        if not has_forbidden:
            adherence_count += 1

        results.append({
            "query": query,
            "response_preview": response_text[:150],
            "tools_called": tools_called,
            "iterations": iterations,
            "completed": completed,
            "relevance_score": round(relevance, 4)
        })

    n = len(test_queries)
    tool_precision = round(appropriate_tool_calls / max(total_tool_calls, 1), 4)
    hallucination_rate = round(hallucination_count / n, 4)
    avg_relevance = round(total_relevance / n, 4)
    task_completion_rate = round(task_complete_count / n, 4)
    prompt_adherence_rate = round(adherence_count / n, 4)

    return {
        "prompt_version": active_version,
        "total_queries": n,
        "metrics": {
            "tool_precision": tool_precision,
            "hallucination_rate": hallucination_rate,
            "response_relevance": avg_relevance,
            "task_completion_rate": task_completion_rate,
            "prompt_adherence_rate": prompt_adherence_rate
        },
        "per_query_results": results
    }

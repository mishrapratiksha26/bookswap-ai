from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List
from app.search import get_similar_books, search_books, get_personal_recommendations
from app.embeddings import generate_embedding
from pymongo import MongoClient
import os
import json
import uuid
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

router = APIRouter()

client = MongoClient(os.environ.get("DB_URL"))
db = client[os.environ.get("DB_NAME", "books")]
books_collection = db["books"]

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# SYSTEM PROMPT — extracted as a constant so Phase 5 (hybrid prompt upgrade)
# and Phase 11 (evaluation: swap v1/v2/v3/v4) can swap this in one place.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are BookSwap AI — a friendly book assistant for students 😊

TOOLS AVAILABLE:
- semantic_search: Search books by topic or keyword
- check_availability: Check if books can be borrowed
- get_alternatives: Find similar books when one is unavailable

RULES:
1. ALWAYS search before recommending. NEVER make up book names.
2. After searching, check availability of ALL found books. Show available books first.
   For unavailable books that are a strong match, still show them with ❌ and try get_alternatives.
3. If a book is unavailable and get_alternatives returns nothing, say "No alternative right now, check back later! 😅"
4. NEVER show book IDs, tool names, JSON, or function calls to the user.
5. NEVER recommend books, courses, or websites not in the inventory.
6. For vague queries ("good books", "easy reads"), ask the user's preferred genre BEFORE searching.
7. Only show books that match the user's request. No DSA books for someone wanting thrillers.
8. For unavailable books, show ❌ and if a returnDate is provided, say "Expected back by [returnDate]".

RESPONSE STYLE:
- Warm and fun, like a helpful librarian 📚
- 1-2 sentence summary per book using its description
- ✅ Available  ❌ Unavailable
- End with a follow-up question 😊"""

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

class AgentResponse(BaseModel):
    response: str
    session_id: str          # echoed back so the frontend can store it
    tools_called: List[str]  # ordered list: e.g. ["semantic_search", "check_availability"]
    iterations: int          # how many loop cycles ran — used in Chapter 5 evaluation

# ---------------------------------------------------------------------------
# TOOL DEFINITIONS
# Three tools defined as JSON schemas. The LLM receives these on every call
# and autonomously decides which to invoke based on its reasoning state.
# Two more tools (get_user_profile, search_pdfs) are added in Phase 6.
# ---------------------------------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "semantic_search",
            "description": "Search BookSwap inventory for books by topic, subject, or keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'beginner data structures'"
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
            "description": "Check if specific books are currently available to borrow on BookSwap",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of book ID strings to check"
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
            "description": "Find similar books to a specific book, useful when a book is unavailable",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_id": {
                        "type": "string",
                        "description": "The book ID to find alternatives for"
                    }
                },
                "required": ["book_id"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# TOOL EXECUTOR
# Maps tool names from the LLM's JSON output to actual Python functions.
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, tool_args: dict):
    if tool_name == "semantic_search":
        books = list(books_collection.find({}))
        results = search_books(tool_args["query"], books, 5)
        return results

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

    return {"error": f"Unknown tool: {tool_name}"}

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

def run_react_loop(
    messages: list,
    available_tools: list,
    max_iterations: int = 7
) -> tuple[str, List[str], int]:

    tools_called = []

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
                return "I'm a bit busy right now 😅 Please try again in a few minutes!", tools_called, iteration
            # Non-rate-limit error: skip this iteration and try again
            continue

        msg = response.choices[0].message

        # No tool calls means LLM has enough information to answer
        if not msg.tool_calls:
            print(f"FINAL: {(msg.content or '')[:200]}")
            return msg.content or "I couldn't find an answer. Please try again.", tools_called, iteration

        # Append assistant message (with its tool_calls) to conversation history
        messages.append(msg)

        # Execute each tool the LLM requested and feed results back
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"TOOL: {tool_name}({tool_args})")
            tools_called.append(tool_name)

            result = execute_tool(tool_name, tool_args)
            print(f"RESULT: {json.dumps(result)[:300]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

    # Safety fallback — should rarely trigger with well-formed queries
    return "Sorry, I wasn't able to complete your request. Please try rephrasing.", tools_called, max_iterations

# ---------------------------------------------------------------------------
# AGENT ENDPOINT
# ---------------------------------------------------------------------------

@router.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest):
    # Assign or preserve session_id — frontend stores this in sessionStorage
    # and sends it back on the next turn for conversation continuity (Phase 8)
    session_id = request.session_id or str(uuid.uuid4())

    # Build the initial messages list.
    # System prompt is first — this is the "instruction layer" the LLM reads
    # before the user's message. Extracted to SYSTEM_PROMPT constant above
    # so Phase 5 (prompt upgrade) and Phase 11 (evaluation) can swap it easily.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.message}
    ]

    response_text, tools_called, iterations = run_react_loop(messages, tools)

    return AgentResponse(
        response=response_text,
        session_id=session_id,
        tools_called=tools_called,
        iterations=iterations
    )

# ---------------------------------------------------------------------------
# EXISTING ENDPOINTS — unchanged
# ---------------------------------------------------------------------------

@router.post("/embed-all")
def embed_all():
    count = 0
    for book in books_collection.find({}):
        text = f"{book.get('title', '')} {book.get('author', '')} {book.get('genre', '')} {book.get('description', '')}"
        vector = generate_embedding(text)
        books_collection.update_one(
            {"_id": book["_id"]},
            {"$set": {"vector": vector}}
        )
        count += 1
    return {"message": f"Embedded {count} books"}

@router.post("/search")
def search(request: SearchRequest):
    books = list(books_collection.find({}))
    results = search_books(request.query, books, request.top_k)
    return {"results": results}

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

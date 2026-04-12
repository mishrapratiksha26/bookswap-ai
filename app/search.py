from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from app.embeddings import generate_embedding


def serialize_doc(doc: dict) -> dict:
    result = {}
    for k, v in doc.items():
        if k == "vector":
            continue
        elif hasattr(v, '__str__') and type(v).__name__ == 'ObjectId':
            result[k] = str(v)
        elif isinstance(v, list):
            result[k] = [
                serialize_doc(i) if isinstance(i, dict) else 
                str(i) if type(i).__name__ == 'ObjectId' else i 
                for i in v
            ]
        elif isinstance(v, dict):
            result[k] = serialize_doc(v)
        else:
            result[k] = v
    return result

# ---------------------------------------------------------------------------
# RE-RANKING FORMULA (Chapter 3.3 — Mathematical Contribution)
#
# Combines five signals into one final score for each candidate book b:
#
#   final_score(b, u, q) = α·sim(q,b) + β·avail(b) + γ·rating(b)
#                        + δ·context(b,u) + ε·pop(b)
#
#   subject to: α + β + γ + δ + ε = 1
#
# Components:
#   sim(q,b)      cosine_similarity(embed(q), embed(b))          ∈ [0,1]
#   avail(b)      1.0 available | 0.3 due ≤7 days | 0.0 unavailable
#   rating(b)     (avg_rating − 1) / 4                           ∈ [0,1]
#                 0.5 if no ratings (neutral cold-start default)
#   context(b,u)  cosine_similarity(taste_vec(u), embed(b))      ∈ [0,1]
#                 0.5 if user has no history (neutral cold-start)
#   pop(b)        log(1 + borrow_count) / log(1 + max_count)     ∈ [0,1]
#                 log-scaled to compress Zipf-distributed borrow counts
#
# Default weights (proposed configuration — ablated in Phase 11):
#   α=0.35  β=0.25  γ=0.15  δ=0.15  ε=0.10
#
# Justification: β (availability) is high because recommending an unavailable
# book with no alternative = wasted interaction on a borrowing platform.
# α (semantic) is highest because relevance is the primary filter.
# δ and ε are secondary personalisation signals.
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "alpha": 0.35,   # semantic similarity
    "beta":  0.25,   # availability
    "gamma": 0.15,   # average rating
    "delta": 0.15,   # user context / taste
    "epsilon": 0.10  # popularity
}

def rerank_books(
    books_with_scores: list,       # output of search_books — has "score" key (sim)
    taste_vector: np.ndarray = None,   # 384-dim user taste vector; None → cold start
    weights: dict = None
) -> list:
    """
    Re-rank candidate books using the 5-component formula.

    Args:
        books_with_scores: list of book dicts, each with a "score" field (cosine sim).
        taste_vector:       precomputed user taste vector (384-dim numpy array).
                            Pass None for new users (cold-start) — context defaults to 0.5.
        weights:            dict with keys alpha/beta/gamma/delta/epsilon.
                            Defaults to DEFAULT_WEIGHTS if None.
    Returns:
        Re-ranked list (descending final_score), same dict shape as input.
    """
    if not books_with_scores:
        return []

    w = weights or DEFAULT_WEIGHTS

    # Pre-compute max borrow_count across candidates for pop normalisation
    max_borrow = max(
        (b.get("borrow_count") or 0 for b in books_with_scores),
        default=1
    )
    if max_borrow == 0:
        max_borrow = 1  # prevent division by zero when all counts are 0

    # Prepare taste vector for context scoring
    if taste_vector is not None:
        taste_arr = np.array([taste_vector])  # shape (1, 384)
    else:
        taste_arr = None

    for book in books_with_scores:
        # --- Component 1: semantic similarity (already computed) ---
        sim = float(book.get("score", 0.0))

        # --- Component 2: availability ---
        if book.get("available") is True:
            avail = 1.0
        elif book.get("available") is False:
            avail = 0.0
        else:
            # None / missing field → treat as available (legacy data)
            avail = 1.0

        # --- Component 3: average rating (normalised to [0,1]) ---
        avg_rating = book.get("avg_rating")
        if avg_rating is not None and avg_rating > 0:
            rating_norm = (float(avg_rating) - 1.0) / 4.0
            rating_norm = max(0.0, min(1.0, rating_norm))
        else:
            rating_norm = 0.5  # neutral cold-start default

        # --- Component 4: user context / taste match ---
        if taste_arr is not None and "vector" in book and book["vector"]:
            try:
                book_arr = np.array([book["vector"]])
                context = float(cosine_similarity(taste_arr, book_arr)[0][0])
                context = max(0.0, min(1.0, context))
            except Exception:
                context = 0.5
        else:
            context = 0.5  # cold-start: neutral for new users

        # --- Component 5: popularity (log-scaled) ---
        borrow_count = book.get("borrow_count") or 0
        pop = math.log1p(borrow_count) / math.log1p(max_borrow)

        # --- Final weighted score ---
        final_score = (
            w["alpha"]   * sim
          + w["beta"]    * avail
          + w["gamma"]   * rating_norm
          + w["delta"]   * context
          + w["epsilon"] * pop
        )
        book["final_score"] = round(final_score, 6)
        # Keep raw components for debugging / ablation analysis
        book["_score_components"] = {
            "sim": round(sim, 4),
            "avail": avail,
            "rating": round(rating_norm, 4),
            "context": round(context, 4),
            "pop": round(pop, 4)
        }

    books_with_scores.sort(key=lambda x: x["final_score"], reverse=True)
    return books_with_scores


def search_books(query: str, books: list, top_k: int = 5) -> list:
    query_vector = generate_embedding(query)
    query_array = np.array([query_vector])

    results = []
    books_with_vectors = 0

    for book in books:
        if "vector" not in book:
            continue
        books_with_vectors += 1
        book_vector = np.array([book["vector"]])
        score = cosine_similarity(query_array, book_vector)[0][0]
        book_data = serialize_doc(book)
        book_data["score"] = float(score)
        results.append(book_data)

    print(f"Total books: {len(books)}, Books with vectors: {books_with_vectors}")
    if results:
        print(f"Top score: {results[0]['score']}")

    results.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] > 0.0]
    return results[:top_k]


def get_similar_books(org_vector, books, book_id, top_k=5) -> list:
    results = []
    org_array = np.array([org_vector])
    for book in books:
        if "vector" not in book:
            continue
        elif str(book["_id"]) == book_id:
            continue
        book_vector = np.array([book["vector"]])
        score = cosine_similarity(org_array, book_vector)[0][0]
        book_data = serialize_doc(book)
        book_data["score"] = float(score)
        results.append(book_data)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] > 0.0]
    return results[:top_k]

def get_personal_recommendations(weighted_books, all_books, library_ids, top_k=5):
    results = []
    weighted_sum = np.zeros(384)
    total_rating = 0
    print(f"weighted_books count: {len(weighted_books)}")
    print(f"library_ids: {library_ids}")
    print(f"all_books count: {len(all_books)}")
    for item in weighted_books:
        vector = np.array(item["vector"])
        rating = item["rating"]
        weighted_sum += vector * rating
        total_rating += rating

    taste_vector = np.array([weighted_sum / total_rating])

    for book in all_books:
        if "vector" not in book:
            continue
        elif str(book["_id"]) in library_ids:
            continue
        book_vector = np.array([book["vector"]])
        score = cosine_similarity(taste_vector, book_vector)[0][0]
        book_data = serialize_doc(book)
        book_data["score"] = float(score)
        results.append(book_data)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] > 0.0]
    return results[:top_k]
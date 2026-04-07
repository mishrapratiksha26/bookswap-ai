from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
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
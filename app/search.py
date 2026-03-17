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

    for book in books:
        if "vector" not in book:
            continue
        book_vector = np.array([book["vector"]])
        score = cosine_similarity(query_array, book_vector)[0][0]
        book_data = serialize_doc(book)
        book_data["score"] = float(score)
        results.append(book_data)

    results.sort(key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] > 0.27]
    return results[:top_k]
from fastapi import APIRouter
from pydantic import BaseModel
from app.search import search_books
from app.embeddings import generate_embedding
from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

client = MongoClient(os.environ.get("DB_URL"))
db = client["test"]
books_collection = db["books"]

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class EmbedRequest(BaseModel):
    book_id: str

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
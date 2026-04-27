from sentence_transformers import SentenceTransformer


# Lazy-load the model. Previously the module-level
#
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#
# caused MiniLM (~90 MB on disk, larger in RAM after loading into a
# torch graph) to be pulled into memory the moment any FastAPI route
# imported app.embeddings — which happens at startup because routes.py
# imports it at the top of the file. Combined with torch's runtime
# overhead and the rest of the process, peak resident size at boot
# pushed past Render's free-plan 512 MB cap and the container was OOM-
# killed before uvicorn finished binding to $PORT (the "No open ports
# detected" symptom in the deploy log).
#
# Lazy initialisation defers the load until the first call to
# generate_embedding(). Startup memory drops by ~150 MB, so the
# container has room to bind the port and accept traffic. The first
# embedding call is slower (cold load), but every subsequent call uses
# the cached singleton — same throughput as before.
_model = None


def _get_model() -> SentenceTransformer:
    """Return the singleton SentenceTransformer instance, loading on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model


def generate_embedding(text: str) -> list:
    embedding = _get_model().encode(text)
    return embedding.tolist()

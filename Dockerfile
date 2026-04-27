# BookSwap FastAPI backend — Docker image
#
# Why Docker on Render (and not the standard Python runtime):
#   The standard Render Python runtime does not have apt-get during the
#   build phase ("Read-only file system" error) — the build container is
#   a non-root sandbox. Tesseract-OCR therefore cannot be installed via
#   apt on that runtime, which means the OCR fallback for scanned
#   lecture-plan PDFs degrades to a "could not read" error on prod even
#   though it works locally on Windows.
#
#   Switching to a Docker runtime gives the build full root access. We
#   apt-install Tesseract once at image-build time, set TESSDATA_PREFIX
#   so PyMuPDF can locate the language data, and OCR works identically
#   on Render and on the developer's laptop. This is the deployment
#   target documented in thesis §6.3 (Limitations) → §6.4 (Further work,
#   resolved as of this build).
#
# Image size budget: the python:3.13-slim base is ~150 MB; tesseract-ocr
# + tesseract-ocr-eng adds ~80 MB; pip install of PyMuPDF +
# sentence-transformers + groq + scikit-learn + numpy resolves to
# ~700 MB more (sentence-transformers downloads MiniLM weights at
# runtime, not build, so they are not in the image). Final image is
# ~950 MB compressed — fits comfortably in Render's free-plan limits.

FROM python:3.13-slim

# --- System dependencies -----------------------------------------------------
# tesseract-ocr           : the OCR engine itself (used via PyMuPDF's
#                           page.get_textpage_ocr() in chapter_extractor.py)
# tesseract-ocr-eng       : English language pack — required for OCR to do
#                           anything; without a language pack tesseract
#                           refuses to run.
# --no-install-recommends : skip suggested packages we don't use, ~60 MB saving
# rm -rf /var/lib/apt/lists : drop apt cache so it doesn't bloat the image
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

# --- Runtime environment -----------------------------------------------------
# TESSDATA_PREFIX  : path to tesseract's language data on Debian Bookworm
#                    (the python:3.13-slim base). The auto-discovery code
#                    in chapter_extractor.py also probes this exact path,
#                    so setting the env var is belt-and-suspenders rather
#                    than strictly required.
# PYTHONUNBUFFERED : send Python's stdout/stderr straight to the Render
#                    log stream without buffering. Without this, our
#                    print(f"[OCR] page N: ...") diagnostics are
#                    invisible until the buffer flushes — diagnostic dead
#                    weight.
# PIP_NO_CACHE_DIR : skip pip's wheel cache. Build is a one-shot; the
#                    cache would only bloat the image with bytes we
#                    never re-use.
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# --- Python dependencies -----------------------------------------------------
# Copy requirements.txt by itself first so that any change to application
# source (the much more common case) does not invalidate the pip-install
# layer. Builds drop from ~7 min to ~30 sec on most code-only changes.
COPY requirements.txt .
RUN pip install --upgrade pip

# Install CPU-only torch FIRST, before requirements.txt resolves it as a
# transitive dependency of sentence-transformers. By default torch on
# PyPI now ships the CUDA build, which pulls in ~2 GB of nvidia-cublas /
# nvidia-cudnn / triton / cuda-toolkit packages. We do CPU inference
# only on Render — the GPU bytes are wasted and they OOM the free
# plan's 512 MB container before uvicorn can bind the port. Pinning to
# PyTorch's CPU wheel index keeps the install ~10× smaller and the
# runtime memory footprint well inside the 512 MB cap. When the
# subsequent `pip install -r requirements.txt` evaluates torch as a
# transitive dep, it sees the CPU build already satisfies the
# constraint and skips the heavyweight CUDA install.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install -r requirements.txt

# --- Application source ------------------------------------------------------
COPY . .

# --- Start command -----------------------------------------------------------
# Render injects $PORT (random per-deploy). Use shell form so the variable
# expands at container start; exec form would treat ${PORT} as a literal
# string. The :-10000 default lets `docker run` work locally without
# setting PORT.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

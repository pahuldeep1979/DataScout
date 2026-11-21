import hashlib
import json
import logging
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from time import perf_counter
from urllib.parse import quote

import pymupdf
from fastapi import APIRouter, FastAPI, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import ROOT, settings
from app.schemas.rag import DocumentInfo, QueryRequest, QueryResponse
from app.services.agent_graph import run_query
from app.services.demo import EXAMPLES, NOTICES, demo_pdf, demo_snapshot
from app.services.hybrid_engine import Workspace, add_embeddings, extract_pdf
from evaluation.run_benchmarks import run_benchmark

logger = logging.getLogger(__name__)
MAX_UPLOAD = 10 * 1024 * 1024
CACHE_LIMIT = 128
workspace = Workspace(settings.INDEX_PATH)
cache = OrderedDict()
cache_lock = Lock()
cache_stats = {"hits": 0, "misses": 0}


@asynccontextmanager
async def lifespan(_app):
    workspace.import_legacy(settings.CHROMA_PERSIST_DIR)
    yield


app = FastAPI(title="DataScout · VeriRAG", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Keep the prototype’s API paths for direct callers and add a same-origin prefix for the UI.
routes = APIRouter()


@routes.get("/health")
def health():
    return {"status": "healthy", "service": "DataScout", "gemini_configured": bool(settings.GEMINI_API_KEY),
            "web_configured": bool(settings.GEMINI_API_KEY and settings.TAVILY_API_KEY)}


@routes.get("/demo")
def get_demo():
    _, documents = demo_snapshot()
    return {"documents": documents, "examples": EXAMPLES}


@routes.get("/demo/documents/{filename}")
def open_demo_pdf(filename: str):
    if filename not in NOTICES:
        raise HTTPException(404, "Demo document not found.")
    return Response(demo_pdf(filename), media_type="application/pdf",
                    headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"})


@routes.post("/documents/upload")
def upload_document(file: UploadFile):
    filename = Path((file.filename or "").replace(chr(92), "/")).name
    if len(filename) > 150 or not filename.lower().endswith(".pdf") or any(ord(c) < 32 for c in filename):
        raise HTTPException(400, "Choose a PDF file with a valid filename.")
    content = file.file.read(MAX_UPLOAD + 1)
    if len(content) > MAX_UPLOAD:
        raise HTTPException(413, "PDF files can be up to 10 MB.")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(400, "The uploaded file is not a readable PDF.")
    try:
        passages, pages = extract_pdf(content, filename)
    except (pymupdf.FileDataError, RuntimeError) as exc:
        raise HTTPException(400, "The PDF could not be read.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    workspace.replace(filename, pages, add_embeddings(passages))
    return {"message": f"Indexed {filename}", "filename": filename, "chunks_indexed": len(passages),
            "page_count": pages}


@routes.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    return workspace.snapshot()[1]


@routes.get("/analytics")
def get_analytics():
    passages, documents = workspace.snapshot()
    with cache_lock:
        hits, misses, entries = cache_stats["hits"], cache_stats["misses"], len(cache)
    return {"status": "success", "live_metrics": {
        "total_documents": len(documents), "total_chunks": len(passages), "cache_entries": entries,
        "cache_hits": hits, "cache_misses": misses, "cache_hit_rate": hits / (hits + misses) if hits + misses else 0,
        "documents": documents,
    }, "benchmarks": run_benchmark()}


@routes.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    start = perf_counter()
    passages, documents = demo_snapshot() if request.workspace == "demo" else workspace.snapshot()
    if set(request.sources) - {doc["filename"] for doc in documents}:
        raise HTTPException(422, "A selected document could not be found.")
    if request.sources:
        passages = [c for c in passages if c.source in request.sources]

    # Cache only the deterministic demo and key it by query, scope and retrieval limit.
    key = hashlib.sha256(json.dumps([
        request.query, request.top_k, sorted(set(request.sources)),
    ]).encode()).hexdigest() if request.workspace == "demo" else None
    with cache_lock:
        cached = cache.get(key)
        if cached:
            cache_stats["hits"] += 1
            cache.move_to_end(key)
        elif key:
            cache_stats["misses"] += 1

    def elapsed() -> float:
        return round((perf_counter() - start) * 1000, 2)

    if cached:
        return cached.model_copy(update={"cache_hit": True, "latency_ms": elapsed(),
                                         "trace_steps": [*cached.trace_steps, "Reused the demo result."]})
    try:
        result = QueryResponse(**run_query(request, passages), latency_ms=elapsed())
    except Exception as exc:
        logger.exception("Evidence workflow failed.")
        raise HTTPException(503, "The evidence workflow was interrupted. Please try again.") from exc
    if key:
        with cache_lock:
            cache[key] = result
            cache.move_to_end(key)
            if len(cache) > CACHE_LIMIT:
                cache.popitem(last=False)
    return result


app.include_router(routes, prefix="/api")
app.include_router(routes, include_in_schema=False)
frontend_dist = ROOT / "frontend" / "dist"
if (frontend_dist / "index.html").exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

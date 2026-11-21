import hashlib
import json
import logging
import re
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.schemas.rag import ChunkResult

logger = logging.getLogger(__name__)
STOP = set("a an and are as at be by can could did do does for from how i in is it me of on or our please should tell that the their this to us was we were what when where which who will with would you your".split())
WORDS = re.compile(r"[^\W_]+(?:[-_][^\W_]+)*", re.UNICODE)
CHUNK_SIZE = 650
MAX_PAGES = 120
MAX_CHUNKS = 1200


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    source: str
    page: int
    text: str
    vector: list[float] | None = None
    model: str = ""


def tokens(text: str) -> list[str]:
    return [token for token in WORDS.findall(text.casefold()) if token not in STOP]


def extract_pdf(data: bytes, filename: str) -> tuple[list[Passage], int]:
    passages = []
    fingerprint = hashlib.sha256(data).hexdigest()[:16]
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if not pdf.is_pdf:
            raise ValueError("The uploaded file must be a PDF.")
        if pdf.needs_pass:
            raise ValueError("Unlock the PDF before uploading it.")
        if len(pdf) > MAX_PAGES:
            raise ValueError(f"PDFs can contain up to {MAX_PAGES} pages.")
        for page_number, page in enumerate(pdf, 1):
            lines = [" ".join(line.split()) for line in page.get_text(sort=True).splitlines()]
            parts = []
            batch = ""
            for line in filter(None, lines):
                if len(batch) + len(line) + 1 > CHUNK_SIZE and batch:
                    parts.append(batch)
                    batch = ""
                if len(line) > CHUNK_SIZE:
                    for start in range(0, len(line), CHUNK_SIZE - 80):
                        parts.append(line[start:start + CHUNK_SIZE])
                        if start + CHUNK_SIZE >= len(line):
                            break
                else:
                    batch = f"{batch}\n{line}" if batch else line
            if batch:
                parts.append(batch)
            if len(passages) + len(parts) > MAX_CHUNKS:
                raise ValueError("This PDF has too much text. Split it into smaller documents.")
            passages.extend(
                Passage(f"{filename}:{fingerprint}:{page_number}:{i}", filename, page_number, part)
                for i, part in enumerate(parts, 1)
            )
        if not passages:
            raise ValueError("No readable text was found. Use a text PDF or run OCR first.")
        return passages, len(pdf)


class Workspace:
    def __init__(self, path: Path):
        self.path = path

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path)
        db.execute("PRAGMA foreign_keys = ON")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                source TEXT PRIMARY KEY, pages INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS passages (
                id TEXT PRIMARY KEY, source TEXT REFERENCES documents(source) ON DELETE CASCADE,
                page INTEGER NOT NULL, text TEXT NOT NULL, vector TEXT, model TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY);
        """)
        return db

    @staticmethod
    def _replace(db, source: str, pages: int, passages: list[Passage]):
        db.execute("DELETE FROM documents WHERE source = ?", (source,))
        db.execute("INSERT INTO documents VALUES (?, ?)", (source, pages))
        db.executemany(
            "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, c.source, c.page, c.text, json.dumps(c.vector) if c.vector else None, c.model)
             for c in passages],
        )

    def replace(self, source: str, pages: int, passages: list[Passage]):
        with closing(self.connect()) as db, db:
            self._replace(db, source, pages, passages)

    def snapshot(self) -> tuple[list[Passage], list[dict]]:
        with closing(self.connect()) as db, db:
            db.execute("BEGIN")
            rows = db.execute("SELECT id, source, page, text, vector, model FROM passages ORDER BY source, page, id")
            passages = [Passage(cid, source, page, text, json.loads(vector) if vector else None, model)
                        for cid, source, page, text, vector, model in rows]
            documents = db.execute("""
                SELECT d.source, COUNT(p.id), d.pages FROM documents d JOIN passages p USING (source)
                GROUP BY d.source ORDER BY d.source
            """)
            return passages, [dict(filename=source, chunk_count=count, page_count=pages)
                              for source, count, pages in documents]

    def import_legacy(self, path: Path):
        if not (path / "chroma.sqlite3").is_file():
            return
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM metadata WHERE key = 'legacy_imported'").fetchone():
                return
            import chromadb
            from chromadb.errors import NotFoundError

            client = chromadb.PersistentClient(path=str(path))
            try:
                collection = client.get_collection("verirag_documents", embedding_function=None)
            except NotFoundError:
                return
            data = collection.get(include=["documents", "metadatas"])
            by_source = {}
            for cid, body, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True):
                source = meta["source"]
                by_source.setdefault(source, []).append(Passage(cid, source, meta["page"], body))
            for source, passages in by_source.items():
                if not db.execute("SELECT 1 FROM documents WHERE source = ?", (source,)).fetchone():
                    self._replace(db, source, max(c.page for c in passages), passages)
            db.execute("INSERT INTO metadata VALUES ('legacy_imported')")


def gemini_client():
    if settings.GEMINI_API_KEY:
        return genai.Client(api_key=settings.GEMINI_API_KEY,
                            http_options=types.HttpOptions(timeout=12_000, retry_options={"attempts": 1}))
    return None


def embeddings(client, texts: list[str], task: str) -> list[list[float]]:
    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL, contents=texts,
        config=types.EmbedContentConfig(task_type=task, output_dimensionality=768),
    )
    vectors = [item.values for item in (response.embeddings or [])]
    if len(vectors) != len(texts) or any(not v or len(v) != 768 for v in vectors):
        raise ValueError("Embedding response did not match the requested inputs.")
    return vectors


def add_embeddings(passages: list[Passage]) -> list[Passage]:
    if not settings.GEMINI_API_KEY:
        return passages
    try:
        with gemini_client() as client:
            vectors = []
            for offset in range(0, len(passages), 100):
                vectors.extend(embeddings(client, [c.text for c in passages[offset:offset + 100]],
                                          "RETRIEVAL_DOCUMENT"))
        return [Passage(c.chunk_id, c.source, c.page, c.text, vector, settings.EMBEDDING_MODEL)
                for c, vector in zip(passages, vectors, strict=True)]
    except Exception:
        logger.warning("Gemini embeddings unavailable. Keeping the local search index.")
        return passages


def term_vector(words: list[str]) -> Counter:
    return Counter(
        [f"word:{word}" for word in words]
        + [f"trigram:{word[i:i + 3]}" for word in words if len(word) >= 4
           for i in range(len(word) - 2)]
    )


def cosine(left: Counter, right: Counter) -> float:
    dot = sum(weight * right[key] for key, weight in left.items())
    if not dot:
        return 0.0
    return dot / ((sum(w * w for w in left.values()) * sum(w * w for w in right.values())) ** 0.5)


class HybridSearch:
    def __init__(self, passages: list[Passage], allow_cloud: bool = False):
        self.passages = passages
        self.allow_cloud = allow_cloud
        self.mode = "local"
        self.corpus = [tokens(chunk.text) for chunk in passages]
        self.vector_corpus = [term_vector(words) for words in self.corpus]
        self.bm25 = BM25Okapi(self.corpus) if any(self.corpus) else None
        if self.bm25:
            # A positive IDF floor keeps exact matches visible even in a one-page library.
            self.bm25.idf = {term: max(score, 0.25) for term, score in self.bm25.idf.items()}

    def ranked(self, query: str, k: int = 5) -> dict[str, list[ChunkResult]]:
        words = tokens(query)
        if not self.bm25 or not words:
            return {mode: [] for mode in ("bm25_only", "vector_only", "hybrid_rrf")}
        lexical = self.bm25.get_scores(list(dict.fromkeys(words)))
        match_vector = term_vector(words)
        vector = [cosine(match_vector, candidate) for candidate in self.vector_corpus]
        self.mode = "local"
        if self.allow_cloud and settings.GEMINI_API_KEY and all(
            c.vector and c.model == settings.EMBEDDING_MODEL for c in self.passages
        ):
            try:
                with gemini_client() as client:
                    query_vector = np.asarray(embeddings(client, [query], "RETRIEVAL_QUERY")[0])
                matrix = np.asarray([c.vector for c in self.passages])
                denom = np.maximum(np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vector), 1e-12)
                vector = matrix @ query_vector / denom
                self.mode = "gemini"
            except Exception:
                logger.warning("Gemini query embedding unavailable. Using local vectors.")

        pool = max(20, k * 2)
        rank_lists = {}
        for name, scores in (("bm25_only", lexical), ("vector_only", vector)):
            ordered = sorted((i for i, score in enumerate(scores) if score > 0),
                             key=lambda i: (-scores[i], self.passages[i].chunk_id))
            rank_lists[name] = {i: rank for rank, i in enumerate(ordered[:pool], 1)}
        bm25_rank, vector_rank = rank_lists.values()
        fused = {
            i: sum(1 / (60 + ranks[i]) for ranks in (bm25_rank, vector_rank) if i in ranks)
            for i in bm25_rank.keys() | vector_rank.keys()
        }
        rank_lists["hybrid_rrf"] = sorted(fused, key=lambda i: (-fused[i], self.passages[i].chunk_id))

        def result(i: int) -> ChunkResult:
            item = self.passages[i]
            return ChunkResult(chunk_id=item.chunk_id, source=item.source, page=item.page, text=item.text,
                               rrf_score=round(fused[i], 6), bm25_rank=bm25_rank.get(i),
                               vector_rank=vector_rank.get(i))

        return {mode: [result(i) for i in list(order)[:k]] for mode, order in rank_lists.items()}

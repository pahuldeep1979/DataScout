from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    source: str
    page: int
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    is_relevant: bool | None = None


class WebSource(BaseModel):
    title: str
    url: str
    snippet: str


class Evidence(BaseModel):
    quote: str
    source: str
    page: int | None = None
    chunk_id: str | None = None
    url: str | None = None
    value: str | None = None


class QueryRequest(BaseModel):
    query: Text = Field(max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    workspace: Literal["demo", "documents"] = "documents"
    sources: list[Text] = Field(default_factory=list, max_length=50)
    public_web_query: Text | None = Field(default=None, max_length=300)


class QueryResponse(BaseModel):
    answer: str
    source_mode: Literal["kb", "web_fallback", "none"]
    verdict: Literal["supported", "conflict", "insufficient"]
    trace_steps: list[str]
    chunks: list[ChunkResult]
    web_sources: list[WebSource]
    evidence: list[Evidence]
    retrieval_mode: Literal["gemini", "local"]
    cache_hit: bool = False
    latency_ms: float = 0


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    page_count: int

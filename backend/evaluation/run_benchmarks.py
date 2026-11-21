"""Reproducible evaluation on the fictional, isolated campus notices."""
import json
from functools import lru_cache
from statistics import mean
from time import perf_counter

import numpy as np

from app.schemas.rag import QueryRequest
from app.services.agent_graph import run_query
from app.services.demo import NOTICES, demo_snapshot
from app.services.hybrid_engine import HybridSearch

CIRCULAR, HANDBOOK, REMINDER = NOTICES
CASES = [
    ("What is the team registration deadline?", [], "conflict", {(CIRCULAR, 1), (REMINDER, 1)}),
    ("When does team registration close?", [], "conflict", {(CIRCULAR, 1), (REMINDER, 1)}),
    ("What is the maximum team size?", [], "supported", {(HANDBOOK, 1)}),
    ("How many people can be in a group?", [], "supported", {(HANDBOOK, 1)}),
    ("Where is the orientation venue?", [], "supported", {(HANDBOOK, 2)}),
    ("What is the team registration deadline?", [CIRCULAR], "supported", {(CIRCULAR, 1)}),
    ("Is there a registration fee waiver?", [], "insufficient", set()),
    ("What is the cancellation policy?", [], "insufficient", set()),
]


@lru_cache
def run_benchmark() -> dict:
    passages, _ = demo_snapshot()
    k = 3
    scores = {mode: [] for mode in ("bm25_only", "vector_only", "hybrid_rrf")}
    routes = []
    latencies = []
    for question, sources, verdict, expected in CASES:
        selected = [c for c in passages if not sources or c.source in sources]
        if expected:
            for mode, results in HybridSearch(selected).ranked(question, k).items():
                pages = list(dict.fromkeys((r.source, r.page) for r in results))
                hits = set(pages) & expected
                rr = next((1 / rank for rank, page in enumerate(pages, 1) if page in expected), 0)
                scores[mode].append((len(hits) / len(expected), len(hits) / k, rr))

        start = perf_counter()
        actual = run_query(QueryRequest(query=question, workspace="demo", sources=sources, top_k=k), selected)
        latencies.append((perf_counter() - start) * 1000)
        cited = {(e.source, e.page) for e in actual["evidence"]}
        routes.append(dict(question=question, scope=sources, expected=verdict, actual=actual["verdict"],
                           correct=verdict == actual["verdict"] and cited == expected))

    return dict(
        k=k, retrieval_cases=sum(bool(case[3]) for case in CASES),
        metrics={mode: dict(zip(("recall", "precision", "mrr"),
                                (round(mean(column), 3) for column in zip(*values))))
                 for mode, values in scores.items()},
        cases=routes, correct=sum(case["correct"] for case in routes),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
    )


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2))

import logging
from urllib.parse import urlsplit

from tavily import TavilyClient

from app.core.config import settings
from app.schemas.rag import WebSource

logger = logging.getLogger(__name__)


def search_web(query: str) -> list[WebSource]:
    try:
        result = TavilyClient(api_key=settings.TAVILY_API_KEY).search(
            query=query, max_results=4, search_depth="basic", timeout=10,
        )
        sources = []
        for item in result.get("results", []):
            url = item.get("url") or ""
            parsed = urlsplit(url)
            snippet = (item.get("content") or "").strip()[:1500]
            if parsed.scheme in ("https", "http") and parsed.hostname and snippet:
                sources.append(WebSource(title=item.get("title") or parsed.hostname, url=url, snippet=snippet))
        return sources
    except Exception:
        logger.warning("Tavily search unavailable.")
        return []

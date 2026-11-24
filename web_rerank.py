"""
src/retrieval/web_rerank.py

Reranks Tavily web search results using the same cross-encoder model
already used in hybrid_retriever.py (ms-marco-MiniLM-L-6-v2), plus a
lightweight domain-authority boost and aggregator filter.

Drop this file into src/retrieval/ and call rerank_web_results()
wherever Tavily results are currently passed straight to the LLM.
"""

from sentence_transformers import CrossEncoder
from urllib.parse import urlparse

# Reuse the same model you already load in hybrid_retriever.py.
# If you already have a CrossEncoder instance there, import and share it
# instead of loading a second copy.
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Domains that should never outrank an authoritative/official source.
# Add known aggregator / low-trust domains here as you discover them.
AGGREGATOR_DOMAINS = {
    "tavily.com",
    "reddit.com",       # example — tune to what you actually see
    "medium.com",
}

# Optional: domains you always want boosted when they match the query
# (e.g. official docs, status pages, the company's own domain).
TRUSTED_DOMAIN_BOOST = 0.15  # added to score if result domain == query entity domain


def _get_domain(url: str) -> str:
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc[4:] if netloc.startswith("www.") else netloc
    except Exception:
        return ""


def filter_aggregators(results: list[dict]) -> list[dict]:
    """Drop results from known low-trust aggregator domains."""
    return [r for r in results if _get_domain(r.get("url", "")) not in AGGREGATOR_DOMAINS]


def rerank_web_results(
    query: str,
    results: list[dict],
    entity_domain: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """
    query: the user's search query
    results: Tavily's raw result list, each a dict with at least
             {"title": ..., "url": ..., "content": ...}
    entity_domain: if you know the canonical domain for the company/entity
                   being searched (e.g. "anthropic.com"), pass it to boost
                   that result above aggregators even if Tavily ranked it lower
    top_k: how many reranked results to keep

    Returns results re-sorted by cross-encoder relevance score (+ boosts),
    highest first.
    """
    if not results:
        return []

    # 1. Drop obvious aggregator noise before scoring
    results = filter_aggregators(results)
    if not results:
        return []

    # 2. Score every (query, passage) pair with the cross-encoder
    pairs = [(query, r.get("content", r.get("title", ""))) for r in results]
    scores = _reranker.predict(pairs)

    for r, score in zip(results, scores):
        boosted = float(score)
        domain = _get_domain(r.get("url", ""))
        if entity_domain and domain == entity_domain.lower():
            boosted += TRUSTED_DOMAIN_BOOST
        r["_rerank_score"] = boosted

    # 3. Sort by final score, descending
    results.sort(key=lambda r: r["_rerank_score"], reverse=True)

    return results[:top_k]


# --- Example usage ---
if __name__ == "__main__":
    # Simulated Tavily response for a query like "Anthropic status page"
    fake_results = [
        {"title": "Some aggregator mirror", "url": "https://tavily.com/x", "content": "..."},
        {"title": "Random blog mention", "url": "https://medium.com/y", "content": "..."},
        {"title": "Anthropic Status", "url": "https://status.anthropic.com", "content": "Official incident history..."},
    ]

    reranked = rerank_web_results(
        query="anthropic incident history",
        results=fake_results,
        entity_domain="anthropic.com",  # note: real domain, used just to illustrate boost matching
    )

    for r in reranked:
        print(f"{r['_rerank_score']:.3f}  {r['url']}")
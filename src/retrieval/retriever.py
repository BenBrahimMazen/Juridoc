"""
Metadata-aware retriever with hybrid re-ranking + multi-domain SOFT BOOST.

Flow: over-fetch a candidate pool CORPUS-WIDE (NO domain exclusion) -> de-dupe to
the best chunk per article -> re-rank by:
    score = ALPHA*embedding_sim + (1-ALPHA)*keyword_overlap + domain_boost
where chunks whose primary OR secondary domain is in the router's top-1 / top-2
predicted domains get a DOMAIN_BOOST / DOMAIN_BOOST_SECOND bump. A wrong router
prediction therefore can never zero out relevant results — it only nudges.

The keyword overlap uses an EXPANDED term set (query_expand) so colloquial
scenario phrasings ("on m'a volé ma voiture") map to formal legal terms ("vol",
"cambriolage", ...) that actually appear in the indexed articles.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

import config
from src.retrieval.query_expand import expand_terms
from src.routing.router import RouteDecision
from src.vectorstore import store

_TOKEN_RE = re.compile(r"[a-zà-ÿ]{4,}")


def _deaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _query_terms(query: str) -> list[str]:
    """Raw query tokens (kept for back-compat / debugging)."""
    return _TOKEN_RE.findall(_deaccent(query).lower())


def _keyword_score(query_terms: list[str], doc_text: str) -> float:
    if not query_terms:
        return 0.0
    hay = _deaccent(doc_text).lower()
    present = sum(1 for t in query_terms if t in hay)
    return present / len(query_terms)


def _boost_domains_from(decision: Any) -> list[str]:
    """Normalize a RouteDecision / legacy domain string into up to 2 boost domains."""
    if isinstance(decision, RouteDecision):
        return [d for d, _ in decision.top_domains[:2]]
    if isinstance(decision, str) and decision and decision != "ambiguous":
        return [decision]
    return []


def retrieve(query: str, decision: Any = None, top_k: int | None = None) -> list[dict[str, Any]]:
    """Corpus-wide over-fetch -> de-dupe -> hybrid re-rank with domain soft boost.

    `decision` is normally a RouteDecision (uses its top-2 domains for boosting),
    but accepts a raw domain string (single-domain boost) or None (no boost) for
    back-compat / user overrides. Retrieval NEVER excludes a domain.
    """
    top_k = top_k or config.TOP_K
    boost = _boost_domains_from(decision)

    pool_size = max(top_k * config.HYBRID_POOL_FACTOR, 20)
    pool = store.query(query, domain=None, top_k=pool_size)  # corpus-wide, no filter
    pool = dedupe_by_article(pool)
    return rerank(query, pool, top_k, boost)


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int,
    boost_domains: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Blend embedding similarity + (expanded) keyword overlap + domain boost."""
    boost_domains = boost_domains or []
    qterms = expand_terms(query)  # expanded colloquial->legal term set
    alpha = config.HYBRID_ALPHA
    b1 = boost_domains[0] if len(boost_domains) >= 1 else None
    b2 = boost_domains[1] if len(boost_domains) >= 2 else None

    scored = []
    for r in results:
        m = r.get("metadata", {}) or {}
        doc = r["text"] + " " + m.get("code_name", "")
        kw = _keyword_score(qterms, doc)
        sim = max(0.0, float(r.get("similarity", 0.0)))
        cd = m.get("domain", "")
        cd2 = m.get("domain_secondary", "")
        boost = 0.0
        if b1 and (cd == b1 or cd2 == b1):
            boost = config.DOMAIN_BOOST
        elif b2 and (cd == b2 or cd2 == b2):
            boost = config.DOMAIN_BOOST_SECOND
        score = alpha * sim + (1.0 - alpha) * kw + boost
        scored.append((score, sim, kw, boost, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, sim, kw, boost, r in scored[:top_k]:
        r2 = dict(r)
        r2["hybrid_score"] = round(score, 4)
        r2["keyword_score"] = round(kw, 4)
        r2["domain_boost"] = round(boost, 4)
        out.append(r2)
    return out


def dedupe_by_article(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse multiple chunks of the same article to the highest-similarity one."""
    seen: dict[tuple, dict[str, Any]] = {}
    for r in results:
        m = r["metadata"]
        key = (m.get("source_file"), m.get("article_number"))
        cur = seen.get(key)
        if cur is None or r["similarity"] > cur["similarity"]:
            seen[key] = r
    return sorted(seen.values(), key=lambda r: r["similarity"], reverse=True)

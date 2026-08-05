"""
Offline multi-domain router (17 domains). Replaces the old binary
finance_banking / general_legal router.

Two cheap signals are blended PER DOMAIN:
1. Keyword hits (domains.KEYWORDS) — strong when the query uses domain vocabulary.
2. Embedding similarity to the domain's exemplar centroid (domains.EXEMPLARS) —
   softer signal for paraphrase / scenario phrasings that lack formal terms.

The router returns a RANKED distribution over all domains + a confidence (top1
margin), NOT a single hard label. Retrieval treats this as a SOFT BOOST and never
excludes a domain — so a wrong prediction can't zero out relevant results.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import config
from src.embeddings.embedder import get_embedding_function
from src.routing.domains import DOMAINS, KEYWORDS, EXEMPLARS


def _keyword_scores(query: str) -> dict[str, float]:
    """Count keyword-phrase hits per domain, normalized to [0,1] by the max."""
    q = query.lower()
    raw = {d: 0 for d in DOMAINS}
    for d in DOMAINS:
        for kw in KEYWORDS[d]:
            if kw in q:
                raw[d] += 1
    mx = max(raw.values()) if raw else 0
    return {d: (v / mx if mx > 0 else 0.0) for d, v in raw.items()}


@dataclass
class RouteDecision:
    primary_domain: str
    top_domains: list[tuple[str, float]]  # ranked (code, combined_score), desc
    confidence: float                     # top1 / (top1 + top2) margin
    reason: str
    scores: dict = field(default_factory=dict)

    # Back-compat alias for older callers that read `.domain`.
    @property
    def domain(self) -> str:
        return self.primary_domain


class Router:
    def __init__(self) -> None:
        self._ef = get_embedding_function()
        self._centroids: dict[str, list[float]] = {}
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        for d in DOMAINS:
            vecs = self._ef.embed_documents(EXEMPLARS[d])
            self._centroids[d] = self._unit_mean(vecs)
        self._ready = True

    @staticmethod
    def _unit_mean(vecs: list[list[float]]) -> list[float]:
        dim = len(vecs[0])
        acc = [0.0] * dim
        for v in vecs:
            for i, x in enumerate(v):
                acc[i] += x
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return [x / norm for x in acc]

    @staticmethod
    def _cos(a: list[float], b: list[float]) -> float:
        # both vectors are L2-normalized -> dot product == cosine
        return sum(x * y for x, y in zip(a, b))

    def _embedding_scores(self, query: str) -> dict[str, float]:
        self._ensure()
        qv = self._ef.embed_query(query)
        raw = {d: max(0.0, self._cos(qv, self._centroids[d])) for d in DOMAINS}
        mx = max(raw.values()) if raw else 0.0
        return {d: (v / mx if mx > 0 else 0.0) for d, v in raw.items()}

    def route(self, query: str) -> RouteDecision:
        kw = _keyword_scores(query)
        emb = self._embedding_scores(query)
        kw_w = config.ROUTER_KW_WEIGHT
        combined = {d: kw_w * kw[d] + (1.0 - kw_w) * emb.get(d, 0.0) for d in DOMAINS}
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
        top_n = max(config.ROUTER_TOP_N, 2)
        top = ranked[:top_n]
        top1 = top[0][1]
        top2 = top[1][1] if len(top) > 1 else 0.0
        confidence = top1 / ((top1 + top2) or 1.0)
        reason = "low_confidence" if confidence < config.ROUTER_LOW_CONF_THRESHOLD else "keyword+embedding"
        return RouteDecision(
            primary_domain=top[0][0],
            top_domains=[(d, round(s, 4)) for d, s in top],
            confidence=round(confidence, 4),
            reason=reason,
            scores={
                "keyword": {d: round(v, 4) for d, v in kw.items()},
                "embedding": {d: round(v, 4) for d, v in emb.items()},
                "combined": {d: round(v, 4) for d, v in combined.items()},
            },
        )


# Default singleton.
_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def route(query: str) -> RouteDecision:
    return get_router().route(query)

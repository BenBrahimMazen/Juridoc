"""
Lightweight evaluation harness for the multi-domain router + retriever.

Checks ROUTRIEVAL correctness (did we pull a chunk from an accepted code?) and
ROUTING correctness against the new 17-domain taxonomy, reported as:
- top-1 routing accuracy (primary domain == expected),
- top-2 routing accuracy (expected in the router's top-2 — the lenient view that
  matches the soft-boost retrieval philosophy),
- retrieval accuracy (top-N chunks include an accepted code_name),
broken out PER DOMAIN and by query STYLE (formal vs scenario).

Optionally also runs end-to-end generation and checks answer structure +
disclaimer when Ollama is up.

Usage:
    python tests/run_eval.py                  # routing + retrieval
    python tests/run_eval.py --generate       # + generation checks (slow)
    python tests/run_eval.py --topn 5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import retriever  # noqa: E402
from src.routing import router as router_mod  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "eval_cases.json"

STRUCTURE_MARKERS = ("Qualification juridique", "Fondement légal",
                     "Réponse", "Nuances", "Recommandation")
DISCLAIMER_MARKERS = ("avocat",)


@dataclass
class CaseResult:
    id: str
    question: str
    style: str
    expected_domain: str
    routed_primary: str = ""
    top_domains: list = field(default_factory=list)
    domain_ok_top1: bool = False
    domain_ok_top2: bool = False
    retrieval_ok: bool = False
    top_codes: list[str] = field(default_factory=list)
    matched_code: str = ""
    gen_ok: bool | None = None
    has_disclaimer: bool | None = None
    structure_score: int | None = None
    answer: str = ""


def _retrieval_correct(results: list[dict], accept_codes: list[str]) -> tuple[bool, str]:
    for r in results:
        cn = (r.get("metadata", {}) or {}).get("code_name", "")
        for sub in accept_codes:
            if sub.lower() in cn.lower():
                return True, cn
    return False, ""


def evaluate(topn: int = 5, generate: bool = False) -> list[CaseResult]:
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    results: list[CaseResult] = []

    llm_on = False
    if generate:
        from src.generation import ollama_client
        llm_on = ollama_client.is_reachable()
        if not llm_on:
            print("[eval] Ollama not reachable -> skipping generation checks.\n")

    for case in cases:
        q = case["question"]
        exp = case["expected_domain"]
        style = case.get("style", "formal")
        decision = router_mod.route(q)
        chunks = retriever.retrieve(q, decision, top_k=max(topn + 2, 8))
        top = chunks[:topn]
        top_codes = [c["metadata"].get("code_name", "") for c in top]

        ret_ok, matched = _retrieval_correct(top, case.get("accept_codes", []))
        top2 = [d for d, _ in decision.top_domains[:2]]
        cr = CaseResult(
            id=case["id"], question=q, style=style, expected_domain=exp,
            routed_primary=decision.primary_domain,
            top_domains=[d for d, _ in decision.top_domains[:3]],
            domain_ok_top1=(decision.primary_domain == exp),
            domain_ok_top2=(exp in top2),
            retrieval_ok=ret_ok, top_codes=top_codes, matched_code=matched,
        )

        if generate and llm_on:
            from src.generation import answerer
            payload = answerer.generate_answer(q, top, llm_enabled=True)
            ans = payload["answer"]
            low = ans.lower()
            cr.answer = ans
            cr.has_disclaimer = any(d in low for d in DISCLAIMER_MARKERS)
            cr.structure_score = sum(1 for m in STRUCTURE_MARKERS if m.lower() in low)
            cr.gen_ok = bool(cr.has_disclaimer and cr.structure_score >= 4)

        results.append(cr)
    return results


def _pct(x: int, n: int) -> float:
    return (x / n * 100.0) if n else 0.0


def _print(results: list[CaseResult], generate: bool) -> None:
    n = len(results)
    print(f"{'ID':<8}{'sty':<4}{'expected':<24}{'routed(top1)':<24}{'T1':<3}{'T2':<3}{'RET':<4}"
          + ("GEN  " if generate else "") + "question")
    print("-" * 116)
    for r in results:
        t1 = "OK" if r.domain_ok_top1 else "."
        t2 = "OK" if r.domain_ok_top2 else "."
        ret = "OK" if r.retrieval_ok else "xx"
        extra = f"{'OK' if r.gen_ok else 'xx':<5}" if generate and r.gen_ok is not None else ""
        print(f"{r.id:<8}{r.style[:3]:<4}{r.expected_domain[:22]:<22}{r.routed_primary[:22]:<22}"
              f"{t1:<3}{t2:<3}{ret:<4}{extra}{r.question[:48]}")
        if not r.retrieval_ok and r.top_codes:
            print(f"              got: {r.top_codes[:3]}")
    print("-" * 116)

    r_top1 = sum(r.domain_ok_top1 for r in results)
    r_top2 = sum(r.domain_ok_top2 for r in results)
    r_ret = sum(r.retrieval_ok for r in results)
    print(f"AGGREGATE ({n} cases)")
    print(f"  Routing top-1 : {_pct(r_top1, n):5.1f}%  ({r_top1}/{n})")
    print(f"  Routing top-2 : {_pct(r_top2, n):5.1f}%  ({r_top2}/{n})")
    print(f"  Retrieval     : {_pct(r_ret, n):5.1f}%  ({r_ret}/{n}, top-{results and len(results[0].top_codes) or 5})")

    # per-domain
    by_dom: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_dom[r.expected_domain].append(r)
    print("\nPER DOMAIN (expected)   cases  top1  top2  ret")
    for d in sorted(by_dom):
        rs = by_dom[d]
        print(f"  {d:<38} {len(rs):>4}  {_pct(sum(x.domain_ok_top1 for x in rs),len(rs)):5.1f}  "
              f"{_pct(sum(x.domain_ok_top2 for x in rs),len(rs)):5.1f}  "
              f"{_pct(sum(x.retrieval_ok for x in rs),len(rs)):5.1f}")

    # per-style
    by_style: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_style[r.style].append(r)
    print("\nPER STYLE   cases  top1  top2  ret")
    for s in sorted(by_style):
        rs = by_style[s]
        print(f"  {s:<12} {len(rs):>4}  {_pct(sum(x.domain_ok_top1 for x in rs),len(rs)):5.1f}  "
              f"{_pct(sum(x.domain_ok_top2 for x in rs),len(rs)):5.1f}  "
              f"{_pct(sum(x.retrieval_ok for x in rs),len(rs)):5.1f}")

    if generate and any(r.gen_ok is not None for r in results):
        gen = [r for r in results if r.gen_ok is not None]
        gen_rate = sum(r.gen_ok for r in gen) / len(gen)
        disc = sum(r.has_disclaimer for r in gen) / len(gen)
        avg_sec = sum(r.structure_score for r in gen) / len(gen)
        print(f"\nGENERATION pass: {gen_rate*100:5.1f}%  ({sum(r.gen_ok for r in gen)}/{len(gen)})")
        print(f"Disclaimer pres.: {disc*100:5.1f}%   Avg sections: {avg_sec:.2f}/5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="also run LLM generation + structure checks")
    ap.add_argument("--topn", type=int, default=5, help="top-N chunks checked for retrieval")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    results = evaluate(topn=args.topn, generate=args.generate)
    if args.json:
        print(json.dumps([{k: v for k, v in r.__dict__.items() if k != "answer"} |
                          {"answer_preview": r.answer[:200]} for r in results],
                         ensure_ascii=False, indent=2))
    else:
        _print(results, args.generate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

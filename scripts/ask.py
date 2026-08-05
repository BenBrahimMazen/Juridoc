"""
End-to-end CLI client: route -> retrieve -> generate, no API server needed.

Usage:
    .venv\\Scripts\\python scripts\\ask.py "Quel est le délai de préavis en cas de licenciement ?"
    .venv\\Scripts\\python scripts\\ask.py "Quelle est la peine pour un chèque sans provision ?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing import router  # noqa: E402
from src.retrieval import retriever  # noqa: E402
from src.generation import answerer, ollama_client  # noqa: E402


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = "Quel est le délai de préavis en cas de licenciement ?"
        print(f"(no question given; using example: {question})\n")

    decision = router.route(question)
    chunks = retriever.retrieve(question, decision)
    llm_on = ollama_client.is_reachable()
    payload = answerer.generate_answer(question, chunks, llm_enabled=llm_on)

    bar = "=" * 88
    print(f"Question   : {question}")
    top_str = ", ".join(f"{d}({s})" for d, s in decision.top_domains[:3])
    print(f"Domaine    : {decision.primary_domain}  (conf={decision.confidence}, {decision.reason})")
    print(f"Top-3      : {top_str}")
    print(f"Ollama     : {'up' if llm_on else 'DOWN (résumé de passages)'}")
    print(f"Récupéré   : {len(chunks)} chunks")
    print(bar)
    print(payload["answer"])
    print(bar)
    print("Sources citées :")
    if payload["citations"]:
        for c in payload["citations"]:
            art = c["article_number"] or "—"
            print(f"  • {c['code_name']}  (art. {art}, sim={c['similarity']})")
    else:
        print("  (aucune)")
    if payload.get("confidence_note"):
        print(f"\nNote : {payload['confidence_note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

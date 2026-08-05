"""
Answer orchestrator: build grounded prompt from retrieved chunks, call the LLM,
then parse [Source N] citations and assemble the API response.

Citations are derived deterministically from the retrieved chunks (real
code_name + article_number + excerpt), mapped via the [Source N] tags the model
was told to emit. This avoids trusting the model to spell out citations.
"""
from __future__ import annotations

import re
from typing import Any

import config
from src.generation import ollama_client, prompts

_SOURCE_TAG_RE = re.compile(r"\[Source\s+(\d+)\]")
_PENALTY_RE = re.compile(
    r"\b(amende|peine|emprisonnement|dinars?|millimes?|TND|sanction(?:s)?|infliger)\b",
    re.IGNORECASE,
)


def _truncate(text: str, n: int = 600) -> str:
    text = text.strip().replace("\n", " ")
    return text if len(text) <= n else text[:n].rstrip() + "…"


def build_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks as a numbered [Source N] block."""
    lines = []
    for i, c in enumerate(chunks, start=1):
        m = c["metadata"]
        ref = f"{m.get('code_name', '?')}"
        art = m.get("article_number", "")
        if art and art != "preamble":
            ref += f", article {art}"
        if m.get("law_reference"):
            ref += f" ({m['law_reference']})"
        # Give the TOP match fuller text so the LLM grounds on the single best
        # article, and trim lower ranks to keep the prompt (and prompt-eval cost)
        # small. Retrieval ranking is unaffected (it runs on full chunk text in the
        # retriever); only the LLM-visible context is tiered.
        n = 1000 if i == 1 else (550 if i == 2 else 250)
        lines.append(f"[Source {i}] {ref}\n{_truncate(c['text'], n)}")
    return "\n\n".join(lines) if lines else "(aucun contexte retrouvé)"


def _mentions_penalty(chunks: list[dict[str, Any]], answer: str) -> bool:
    if _PENALTY_RE.search(answer):
        return True
    return any(_PENALTY_RE.search(c["text"]) for c in chunks)


def _build_citations(
    answer: str, chunks: list[dict[str, Any]]
) -> list[dict[str, str]]:
    cited = set(int(n) for n in _SOURCE_TAG_RE.findall(answer))
    if not cited:
        # Model didn't tag sources — fall back to the top few retrieved chunks.
        cited = set(range(1, min(4, len(chunks)) + 1))
    out = []
    for idx in sorted(cited):
        if 1 <= idx <= len(chunks):
            c = chunks[idx - 1]
            m = c["metadata"]
            out.append({
                "code_name": m.get("code_name", ""),
                "article_number": m.get("article_number", ""),
                "source_file": m.get("source_file", ""),
                "law_reference": m.get("law_reference", ""),
                "excerpt": _truncate(c["text"], 320),
                "similarity": round(c.get("similarity", 0.0), 4),
            })
    # Dedupe by (source_file, article_number).
    seen, dedup = set(), []
    for cit in out:
        k = (cit["source_file"], cit["article_number"])
        if k not in seen:
            seen.add(k)
            dedup.append(cit)
    return dedup


def _confidence_note(
    chunks: list[dict[str, Any]], answer: str, llm_ok: bool, low_sim: bool
) -> str | None:
    notes = []
    if not llm_ok:
        notes.append(
            "Le modèle de génération local (Ollama) n'était pas joignable ; "
            "la réponse ci-dessous est un résumé des passages pertinents "
            "retrouvés et non une rédaction finale."
        )
    if low_sim or not chunks:
        notes.append(prompts.INSUFFICIENT_NOTE)
    if _mentions_penalty(chunks, answer):
        notes.append(prompts.PENALTY_NOTE)
    return " ".join(notes) if notes else None


def _best_similarity(chunks: list[dict[str, Any]]) -> float:
    return max((c.get("similarity", 0.0) for c in chunks), default=0.0)


def finalize(chunks: list[dict[str, Any]], answer: str, *, llm_ok: bool) -> dict[str, Any]:
    """Derive citations + confidence note from a finished answer text.

    Split out from :func:`generate_answer` so a streaming caller can accumulate
    the answer itself, then hand the full text here to get the same deterministic
    citations/notes (mapped from the ``[Source N]`` tags in the answer).
    """
    low_sim = _best_similarity(chunks) < 0.45
    return {
        "citations": _build_citations(answer, chunks) if chunks else [],
        "confidence_note": _confidence_note(chunks, answer, llm_ok=llm_ok, low_sim=low_sim),
    }


def generate_answer(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    llm_enabled: bool = True,
) -> dict[str, Any]:
    """Produce the final answer payload from a question + retrieved chunks."""
    # No grounding at all -> short, honest message, no LLM call needed.
    if not chunks:
        return {
            "answer": (
                "Les documents indexés ne contiennent pas d'élément suffisant "
                "pour répondre à cette question. Je vous recommande de consulter "
                "un avocat inscrit pour examiner votre situation.\n\n"
                "Avertissement : informations juridiques générales à titre "
                "indicatif ; la consultation d'un avocat est recommandée pour "
                "votre cas particulier."
            ),
            "citations": [],
            "confidence_note": prompts.INSUFFICIENT_NOTE,
        }

    context = build_context(chunks)
    user = prompts.USER_HEADER.format(question=question.strip(), context=context)

    llm_ok = False
    if llm_enabled:
        try:
            answer = ollama_client.chat(prompts.SYSTEM_PROMPT, user)
            llm_ok = bool(answer)
        except Exception as e:
            answer = _fallback_summary(question, chunks) + f"\n\n[erreur LLM: {e}]"
    else:
        answer = _fallback_summary(question, chunks)

    fin = finalize(chunks, answer, llm_ok=llm_ok)
    return {
        "answer": answer,
        "citations": fin["citations"],
        "confidence_note": fin["confidence_note"],
    }


def _fallback_summary(question: str, chunks: list[dict[str, Any]]) -> str:
    """Used when the LLM is unavailable: synthesise a transparent excerpt-based answer."""
    parts = [
        "Réponse provisoire (le modèle de génération est indisponible) fondée "
        "sur les passages les plus pertinents retrouvés :"
    ]
    for i, c in enumerate(chunks[:4], start=1):
        m = c["metadata"]
        art = f", article {m['article_number']}" if m.get("article_number") and m["article_number"] != "preamble" else ""
        parts.append(f"\n[Source {i}] {m.get('code_name', '')}{art}:\n{_truncate(c['text'], 500)}")
    parts.append(
        "\n\nAvertissement : informations juridiques générales à titre indicatif ; "
        "la consultation d'un avocat inscrit est recommandée pour votre cas particulier."
    )
    return "\n".join(parts)

"""
Resolve a human-readable `code_name` for each source PDF.

The filename→code_name table now lives in `src/ingestion/file_meta.py` (the
single source of truth, which also carries each file's domain). It is mirrored
here as `CODE_NAME_OVERRIDES` so the first-page title-extraction heuristics below
can fall back to it, and so `scripts/refresh_code_names.py` keeps working.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from src.ingestion.file_meta import FILE_META

# filename -> code_name, derived from FILE_META (do not edit here — edit file_meta.py).
CODE_NAME_OVERRIDES: dict[str, str] = {fn: m["code_name"] for fn, m in FILE_META.items()}

_STOPWORDS = {
    "tunisie", "republique", "république", "ministere", "ministère", "journal",
    "officiel", "page", "tunisien", "fr", "www", "http", "pdf",
}
_BRACKETS_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_NONALNUM_RE = re.compile(r"[^A-Za-zÀ-ÿ0-9'\- ]+")


def _clean_stem(stem: str) -> str:
    stem = stem.replace("_", " ").replace("-", " ").strip()
    stem = _BRACKETS_RE.sub("", stem)
    stem = _NONALNUM_RE.sub("", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def extract_title_from_first_page(first_page_text: str) -> Optional[str]:
    """Best-effort: pull a plausible document title from the first page."""
    if not first_page_text:
        return None
    lines = [ln.strip() for ln in first_page_text.splitlines() if ln.strip()]
    candidates: list[str] = []
    for ln in lines[:25]:
        low = ln.lower()
        if len(ln) < 4 or len(ln) > 120:
            continue
        if any(sw in low for sw in _STOPWORDS) and len(ln) < 25:
            continue
        if re.search(r"@\d|\d{4,}", ln):
            continue
        candidates.append(ln)
        if len(candidates) >= 6:
            break
    if not candidates:
        return None
    candidates.sort(key=lambda s: (-len(s), lines.index(s) if s in lines else 99))
    title = candidates[0]
    title = re.sub(r"\s+", " ", title).strip(" .-")
    return title if len(title) >= 5 else None


def resolve_code_name(pdf_path: Path, first_page_text: str = "") -> str:
    """Return the best code_name for a PDF, applying the override->title->stem chain."""
    name = pdf_path.name
    if name in CODE_NAME_OVERRIDES:
        return CODE_NAME_OVERRIDES[name]
    title = extract_title_from_first_page(first_page_text)
    if title:
        return title
    return _clean_stem(pdf_path.stem) or name

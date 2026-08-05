"""
Ingestion orchestration: PDF -> per-article chunks -> embeddings -> ChromaDB.

Domain (+ optional secondary domain) and code_name are resolved PER-FILE from
`src/ingestion/file_meta.py` (the single source of truth), NOT from the folder.

Public entry points:
    ingest_file(pdf_path, domain=None)
    ingest_domain(domain_code)      # all on-disk files whose FILE_META domain == code
    ingest_files(pdf_paths)         # arbitrary iterable (e.g. a single folder)
    ingest_all()                    # every on-disk source PDF known to FILE_META
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import config
from src.ingestion.chunker import chunk_by_articles, Chunk
from src.ingestion.code_names import resolve_code_name
from src.ingestion.file_meta import get_file_meta
from src.ingestion.pdf_parser import extract_pdf
from src.routing.domains import DOMAIN_SET
from src.vectorstore import store

# Best-effort detection of a promulgating reference (Loi / Décret n° ...).
_LAW_REF_RE = re.compile(
    r"(?:Loi(?:\s+organique)?|D[ée]cret|Ordonnance)\s+n[°o]\s*"
    r"(\d{2,4}[-–]?\d{1,4})(?:\s+du\s+\d{1,2}\s+\w+\s+\d{4})?",
    re.IGNORECASE,
)


def _detect_law_reference(*texts: str) -> str:
    for t in texts:
        if not t:
            continue
        m = _LAW_REF_RE.search(t)
        if m:
            return m.group(0).strip()
    return ""


def _build_chunk_dicts(
    chunks: list[Chunk],
    *,
    code_name: str,
    domain: str,
    domain_secondary: str,
    law_reference: str,
) -> list[dict]:
    return [
        {
            "text": c.text,
            "article_number": c.article_number,
            "code_name": code_name,
            "domain": domain,
            "domain_secondary": domain_secondary,
            "law_reference": law_reference,
        }
        for c in chunks
    ]


def _all_source_pdfs() -> list[Path]:
    """Every *.pdf under existing source dirs, de-duplicated by filename."""
    out: list[Path] = []
    seen: set[str] = set()
    for d in config.source_dirs():
        for p in sorted(d.glob("*.pdf")):
            if p.name not in seen:
                seen.add(p.name)
                out.append(p)
    return out


def ingest_file(pdf_path: Path, domain: str | None = None, force_ocr: bool = False) -> dict:
    """Ingest a single PDF. Domain/code_name/secondary resolved from FILE_META.

    `domain` overrides FILE_META if given. `force_ocr` ignores a corrupted text
    layer and OCRs every page. Files not in FILE_META (e.g. the excluded Moroccan
    code) return status "skipped" and are not ingested.
    """
    pdf_path = Path(pdf_path)
    meta = get_file_meta(pdf_path.name)
    domain = domain or meta["domain"]
    if not domain:
        return {
            "file": pdf_path.name,
            "status": "skipped",
            "error": "not in FILE_META (no domain mapping)",
        }

    parsed = extract_pdf(pdf_path, force_ocr=force_ocr)
    code_name = meta["code_name"] or resolve_code_name(pdf_path, parsed.first_page_text)
    law_reference = _detect_law_reference(
        code_name, parsed.first_page_text, parsed.text[:4000]
    )
    domain_secondary = meta["domain_secondary"]

    chunks = chunk_by_articles(
        parsed.text,
        max_chars=config.SUBCHUNK_MAX_CHARS,
        min_chars=config.SUBCHUNK_MIN_CHARS,
    )
    chunk_dicts = _build_chunk_dicts(
        chunks,
        code_name=code_name,
        domain=domain,
        domain_secondary=domain_secondary,
        law_reference=law_reference,
    )
    written = store.replace_source_chunks(pdf_path.name, chunk_dicts)

    return {
        "file": pdf_path.name,
        "domain": domain,
        "domain_secondary": domain_secondary,
        "code_name": code_name,
        "law_reference": law_reference,
        "pages": parsed.n_pages,
        "pages_with_text": parsed.n_pages_with_text,
        "chunks_written": written,
        "status": "ok" if written else ("empty" if not parsed.text else "no_chunks"),
    }


def ingest_domain(domain: str) -> list[dict]:
    if domain not in DOMAIN_SET:
        raise ValueError(f"unknown domain: {domain}")
    pdfs = [p for p in _all_source_pdfs() if get_file_meta(p.name)["domain"] == domain]
    return _run(pdfs)


def ingest_files(pdf_paths: Iterable[Path]) -> list[dict]:
    return _run([Path(p) for p in pdf_paths])


def ingest_all() -> list[dict]:
    return _run(_all_source_pdfs())


def _run(pdfs: list[Path]) -> list[dict]:
    results: list[dict] = []
    try:
        from tqdm import tqdm
        iterator = tqdm(pdfs, desc="ingest", unit="pdf")
    except Exception:  # tqdm optional
        iterator = pdfs

    for pdf_path in iterator:
        try:
            results.append(ingest_file(pdf_path))
        except Exception as e:  # don't let one bad PDF abort the whole run
            results.append({
                "file": Path(pdf_path).name,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
            })
    return results


def summary(results: list[dict]) -> dict:
    from collections import Counter
    ok = [r for r in results if r.get("status") == "ok"]
    by_domain = Counter(r.get("domain") for r in ok if r.get("domain"))
    return {
        "total_files": len(results),
        "ok": len(ok),
        "skipped": len([r for r in results if r.get("status") == "skipped"]),
        "errors": len([r for r in results if r.get("status") == "error"]),
        "empty": len([r for r in results if r.get("status") in ("empty", "no_chunks")]),
        "total_chunks_written": sum(r.get("chunks_written", 0) for r in results),
        "indexed_chunks_total": store.count(),
        "by_domain": dict(by_domain),
    }

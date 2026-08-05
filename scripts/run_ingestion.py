"""
CLI entry point for ingestion.

Domain/code_name/secondary are resolved PER-FILE from src/ingestion/file_meta.py.

Usage (run with the project venv):
    python scripts/run_ingestion.py                            # all known source PDFs
    python scripts/run_ingestion.py droit_penal                # one 17-domain bucket
    python scripts/run_ingestion.py --file "023.pdf"           # one source file
    python scripts/run_ingestion.py --dir "pdfs/new"           # every PDF in a folder
    python scripts/run_ingestion.py --file "loi2001-36fr.pdf" --force-ocr

Idempotent: re-running replaces each source file's chunks cleanly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.ingestion import pipeline  # noqa: E402
from src.routing.domains import DOMAINS  # noqa: E402


def _find_file(name: str) -> Path | None:
    for d in config.source_dirs():
        cand = d / name
        if cand.exists():
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Tunisian legal PDFs into ChromaDB.")
    ap.add_argument("domain", nargs="?", choices=list(DOMAINS),
                    help="Ingest only this 17-domain bucket (default: all source PDFs).")
    ap.add_argument("--file", help="Ingest a single source file name only.")
    ap.add_argument("--dir", help="Ingest every PDF in this directory.")
    ap.add_argument("--force-ocr", action="store_true",
                    help="Ignore the text layer and OCR every page (for corrupted-text PDFs).")
    args = ap.parse_args()

    if args.file:
        target = _find_file(args.file)
        if not target:
            print(f"file not found: {args.file}", file=sys.stderr)
            return 2
        results = [pipeline.ingest_file(target, force_ocr=args.force_ocr)]
    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"directory not found: {args.dir}", file=sys.stderr)
            return 2
        results = pipeline.ingest_files(sorted(d.glob("*.pdf")))
    elif args.domain:
        results = pipeline.ingest_domain(args.domain)
    else:
        results = pipeline.ingest_all()

    print("\n=== Per-file results ===")
    for r in results:
        st = r.get("status")
        if st == "ok":
            print(f"  [OK]   {r['file']:<55} chunks={r['chunks_written']:<5} "
                  f"pages={r['pages_with_text']}/{r['pages']}  dom={r['domain']}  ({r['code_name'][:40]})")
        elif st in ("empty", "no_chunks"):
            print(f"  [WARN] {r['file']:<55} status={st}")
        elif st == "skipped":
            print(f"  [SKIP] {r['file']:<55} {r.get('error', st)}")
        else:
            print(f"  [ERR]  {r['file']:<55} {r.get('error', st)}")

    summ = pipeline.summary(results)
    print("\n=== Summary ===")
    print(json.dumps(summ, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

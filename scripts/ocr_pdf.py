"""Resumable OCR of a scanned PDF to a text cache, page by page.

Writes data/processed/ocr_cache/<stem>.txt (one [[PAGE n]] block per page) and
resumes from the last completed page on re-run, so interruptions / machine sleep
don't lose progress. Bounded with --max-pages so it fits in a short foreground run.

Usage:
    .venv\\Scripts\\python scripts\\ocr_pdf.py "Loi_2016_48_fr.pdf"                 # all remaining
    .venv\\Scripts\\python scripts\\ocr_pdf.py "Loi_2016_48_fr.pdf" --max-pages 12  # bounded chunk
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # noqa: E402
import config  # noqa: E402
from src.ingestion.pdf_parser import _ocr_page  # noqa: E402

CACHE_DIR = config.PROCESSED_DIR / "ocr_cache"
_PAGE_MARK = re.compile(r"\[\[PAGE \d+\]\]")


def find_pdf(name: str) -> Path:
    for d in ("finance_banking", "general_legal"):
        for p in config.raw_dirs(d):
            cand = p / name
            if cand.exists():
                return cand
    raise FileNotFoundError(name)


def pages_done(cache_file: Path) -> int:
    if not cache_file.exists():
        return 0
    return len(_PAGE_MARK.findall(cache_file.read_text(encoding="utf-8", errors="ignore")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = all remaining")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    pdf = find_pdf(args.file)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (pdf.stem + ".txt")
    doc = fitz.open(pdf)
    total = doc.page_count
    done = pages_done(cache)
    remaining = total - done
    limit = remaining if args.max_pages <= 0 else min(remaining, args.max_pages)
    print(f"{pdf.name}: {total} pages, {done} cached -> OCR-ing {limit} from page {done + 1}")

    with cache.open("a", encoding="utf-8") as f:
        for k in range(limit):
            i = done + k
            t = _ocr_page(doc.load_page(i), dpi=args.dpi)
            f.write(f"\n\n[[PAGE {i + 1}]]\n{t}\n")
            f.flush()
            print(f"  page {i + 1}/{total}: {len(t)} chars", flush=True)
    doc.close()

    done2 = pages_done(cache)
    print(f"now {done2}/{total} pages cached" + ("  [COMPLETE]" if done2 >= total else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

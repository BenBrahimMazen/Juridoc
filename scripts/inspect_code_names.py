"""Dump first-page text for each PDF so opaque filenames can be given real code_names.

Run:  .venv\\Scripts\\python scripts\\inspect_code_names.py
Then edit src/ingestion/code_names.py CODE_NAME_OVERRIDES and re-ingest the files.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.ingestion.pdf_parser import extract_pdf  # noqa: E402


def main() -> int:
    for domain in ("finance_banking", "general_legal"):
        for d in config.raw_dirs(domain):
            for pdf in sorted(d.glob("*.pdf")):
                try:
                    p = extract_pdf(pdf)
                    snippet = " ".join(p.first_page_text.split())[:320]
                except Exception as e:
                    snippet = f"<error: {e}>"
                print(f"\n[{domain}] {pdf.name}")
                print(f"    pages={p.n_pages}  ->  {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

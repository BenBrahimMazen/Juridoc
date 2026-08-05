"""
Full index inventory: chunks/articles per file, grouped by primary domain, with
secondary-domain tags. Read-only. Used for the deliverable report.

Usage:  python scripts/inventory.py
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.routing.domains import DOMAINS, LABELS  # noqa: E402
from src.vectorstore import store  # noqa: E402

# filename -> {code_name, domain, domain_secondary} (single source of truth)
from src.ingestion.file_meta import FILE_META  # noqa: E402


def main() -> int:
    coll = store.get_collection()
    res = coll.get(include=["metadatas"])
    metas = res.get("metadatas") or []

    print("=" * 100)
    print(f"INDEX TOTAL: {len(metas)} chunks across {len({m.get('source_file') for m in metas})} files")
    dom = Counter(m.get("domain", "?") for m in metas)
    print("by primary domain:")
    for d in DOMAINS:
        if dom.get(d):
            print(f"   {dom[d]:>5}  {d}  ({LABELS[d]})")
    other = {k: v for k, v in dom.items() if k not in DOMAINS}
    if other:
        print("   UNMAPPED:", other)
    sec = sum(1 for m in metas if m.get("domain_secondary"))
    print(f"chunks carrying a secondary domain tag: {sec}")

    # per-file
    files = defaultdict(lambda: {"domain": "?", "domain_secondary": "", "code": "?",
                                 "chunks": 0, "articles": set()})
    for m in metas:
        f = m.get("source_file", "?")
        d = files[f]
        d["domain"] = m.get("domain", "?")
        d["domain_secondary"] = m.get("domain_secondary", "") or ""
        d["code"] = m.get("code_name", "?")
        d["chunks"] += 1
        if m.get("article_number"):
            d["articles"].add(m["article_number"])

    # files present in FILE_META but NOT indexed (pending ingestion / excluded)
    indexed = set(files.keys())
    pending = [fn for fn in FILE_META if fn not in indexed]
    excluded = [fn for fn in ("Code-des-Juridictions-Financieres.pdf",) if fn not in indexed]

    print("\n" + "=" * 100)
    print("PER FILE (by primary domain)")
    print("-" * 100)
    for d in DOMAINS:
        fs = [f for f in files if files[f]["domain"] == d]
        if not fs:
            continue
        print(f"\n## {d}  ({LABELS[d]})  — {len(fs)} file(s), {sum(files[f]['chunks'] for f in fs)} chunks")
        for f in sorted(fs):
            fd = files[f]
            secstr = f"  [2nd: {fd['domain_secondary']}]" if fd["domain_secondary"] else ""
            print(f"   {fd['chunks']:>5} chk / {len(fd['articles']):>4} art  {f}{secstr}")
            print(f"            {fd['code']}")

    if pending:
        print("\n" + "=" * 100)
        print("FILE_META entries NOT YET INDEXED (pending ingestion):")
        for fn in pending:
            m = FILE_META[fn]
            print(f"   {fn[:60]:<60} -> {m['domain']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

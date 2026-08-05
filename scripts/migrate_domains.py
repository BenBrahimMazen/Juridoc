"""
Re-tag the domain (+ domain_secondary) metadata on EXISTING chunks to the new
17-domain taxonomy, using FILE_META as the source of truth.

In-place via `coll.update(ids, metadatas)` — NO re-embedding (fast, idempotent).
Run after editing src/ingestion/file_meta.py; safe to re-run (only updates
chunks whose domain/secondary differ).

Usage:
    .venv\\Scripts\\python scripts\\migrate_domains.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.file_meta import FILE_META  # noqa: E402
from src.vectorstore import store  # noqa: E402


def main() -> int:
    coll = store.get_collection()
    res = coll.get(include=["metadatas"])
    ids = res.get("ids", []) or []
    metas = res.get("metadatas", []) or []

    up_ids, up_metas = [], []
    missing: set[str] = set()
    for _id, meta in zip(ids, metas):
        sf = (meta or {}).get("source_file", "")
        m = FILE_META.get(sf)
        if not m:
            missing.add(sf)
            continue
        new_domain = m["domain"]
        new_sec = m.get("domain_secondary", "")
        cur = dict(meta or {})
        if cur.get("domain") == new_domain and cur.get("domain_secondary", "") == new_sec:
            continue
        cur["domain"] = new_domain
        cur["domain_secondary"] = new_sec
        up_ids.append(_id)
        up_metas.append(cur)

    print(f"total chunks: {len(ids)}")
    print(f"chunks to update: {len(up_ids)}")
    if missing:
        print(f"WARN: {len(missing)} chunks belong to files NOT in FILE_META (left as-is):")
        for sf in sorted(missing):
            print(f"   - {sf}")
    if not up_ids:
        print("nothing to update (all domains already match FILE_META).")
        return 0

    B = 5000
    for i in range(0, len(up_ids), B):
        coll.update(ids=up_ids[i:i + B], metadatas=up_metas[i:i + B])
    print(f"updated {len(up_ids)} chunks in place (no re-embedding).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

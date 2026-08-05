"""Update code_name metadata on already-indexed chunks without re-embedding.

Run after editing src/ingestion/code_names.py:
    .venv\\Scripts\\python scripts\\refresh_code_names.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.code_names import CODE_NAME_OVERRIDES  # noqa: E402
from src.vectorstore import store  # noqa: E402


def main() -> int:
    coll = store.get_collection()
    res = coll.get(include=["metadatas"])
    ids = res.get("ids", []) or []
    metas = res.get("metadatas", []) or []
    print(f"indexing {len(ids)} chunks...")

    up_ids, up_metas, by_file = [], [], {}
    for _id, meta in zip(ids, metas):
        sf = (meta or {}).get("source_file", "")
        new_name = CODE_NAME_OVERRIDES.get(sf)
        if not new_name or (meta or {}).get("code_name") == new_name:
            continue
        m = dict(meta)
        m["code_name"] = new_name
        up_ids.append(_id)
        up_metas.append(m)
        by_file[sf] = new_name

    if not up_ids:
        print("nothing to update (all code_names already match overrides).")
        return 0

    # Chroma update in batches of 5000.
    B = 5000
    for i in range(0, len(up_ids), B):
        coll.update(ids=up_ids[i:i + B], metadatas=up_metas[i:i + B])

    print(f"\nupdated {len(up_ids)} chunks across {len(by_file)} files:")
    for sf, name in sorted(by_file.items()):
        print(f"  {sf:<58} -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

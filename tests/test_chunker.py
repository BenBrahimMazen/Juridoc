"""
Unit tests for the per-article chunker. Runnable with pytest OR as a script:

    .venv\\Scripts\\python tests\\test_chunker.py
    # or, after `pip install pytest`:
    .venv\\Scripts\\python -m pytest tests\\test_chunker.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import chunk_by_articles, _normalize_num  # noqa: E402


def _art_nums(text, **kw):
    return [c.article_number for c in chunk_by_articles(text, **kw) if c.is_article]


def test_basic_articles_split():
    text = ("Préambule du code.\n\n"
            "Article 1\nPremier article ici.\n\n"
            "Article 2\nDeuxième article ici.\n")
    nums = _art_nums(text)
    assert nums == ["1", "2"], nums


def test_article_premier_normalizes_to_1():
    assert _normalize_num("premier") == "1"
    assert _normalize_num("1er") == "1"
    text = "Article premier\nTexte.\nArticle 2\nSuite.\n"
    assert _art_nums(text)[0] == "1"


def test_bis_ter_suffixes_preserved():
    assert _normalize_num("5 bis") == "5 bis"
    assert _normalize_num("12 ter") == "12 ter"
    text = ("Article 5 bis\nTexte.\n\n"
            "Art. 12 ter -\nTexte.\n\n"
            "Art.254.- Texte.\n")
    nums = _art_nums(text)
    assert "5 bis" in nums and "12 ter" in nums and "254" in nums, nums


def test_art_abbreviation_variants():
    text = ("Article 1\nx\n\nArt. 2\nx\n\nArt.3 -\nx\n\nArt 4.-\nx\n")
    nums = _art_nums(text)
    assert nums == ["1", "2", "3", "4"], nums


def test_midsentence_article_not_split():
    # "l'article 5" mid-line must NOT create a boundary.
    text = ("Article 1\nConformément à l'article 5 du code, la règle s'applique.\n\n"
            "Article 2\nFin.\n")
    nums = _art_nums(text)
    assert nums == ["1", "2"], nums


def test_preamble_chunk_emitted():
    text = ("TITRE PRÉLIMINAIRE\nDispositions générales.\n\n"
            "Article 1\nTexte.\n")
    chunks = chunk_by_articles(text)
    pre = [c for c in chunks if not c.is_article]
    assert len(pre) == 1
    assert "TITRE PRÉLIMINAIRE" in pre[0].text


def test_long_article_subchunked_with_same_number():
    para = "Paragraphe de contenu juridique suffisamment long pour dépasser. " * 12
    body = "\n\n".join(para for _ in range(6))
    text = f"Article 42\n{body}\nArticle 43\nCourt.\n"
    chunks = chunk_by_articles(text, max_chars=400, min_chars=80)
    a42 = [c for c in chunks if c.article_number == "42"]
    assert len(a42) >= 2, len(a42)
    assert all(c.chunk_index == i for i, c in enumerate(a42))
    # Sub-chunks keep the article number in metadata-ready form.
    assert all(c.article_number == "42" for c in a42)


def test_fallback_when_not_article_structured():
    text = "Ceci est un texte sans numérotation d'articles du tout.\n\n" \
           "Juste des paragraphes.\n\n" * 10
    chunks = chunk_by_articles(text)
    assert chunks, "expected fallback chunks"
    assert all(not c.is_article for c in chunks)
    assert all(c.article_number == "preamble" for c in chunks)


def test_empty_text_returns_empty():
    assert chunk_by_articles("") == []
    assert chunk_by_articles("   \n\n  ") == []


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)

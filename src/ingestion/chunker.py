"""
Per-article chunking for French Tunisian legal texts.

An "article" boundary is recognised by French legal numbering conventions:
    Article 254 / Article premier / Article 5 bis
    Art. 254 / Art.254 / Art. 254 - / Art. 254.-

Each chunk carries the article number in metadata. Very long articles are
sub-chunked on paragraph boundaries (never mid-sentence), and every sub-chunk
keeps the article number + a chunk index.

Amendment annotations ("Modifié par la loi n°…", "Abrogé par…") live inside the
article text and are preserved verbatim — they are legally relevant context.

If a PDF yields too few article boundaries (some "recueils" or lois are
structured by chapter/section only), we fall back to paragraph-based chunking so
the text is still indexed and citable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Ordinal suffixes used in French legal numbering.
_SUFFIXES = (
    r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?"
)
# "premier"/"première"/"1er"/"1ère" -> normalised to "1".
_PREMIER = r"(?:premier|première|1er|1ère)"
_NUM = rf"(?:{_PREMIER}|[0-9]{{1,4}}){_SUFFIXES}"

# Matches an article header. Must be at a line start (start-of-text or after \n)
# to avoid catching the word "article" mid-sentence. Case-insensitive so all-caps
# headers ("ARTICLE 5") also match. Handles "Article 5", "Art. 5", "Art.5",
# "Art 5", "Article premier", "Article 5 bis".
ARTICLE_HEADER_RE = re.compile(
    rf"(?:^|\n)[ \t]*(Art(?:icle)?\.?\s*({_NUM}))\b",
    re.IGNORECASE,
)

# Whitespace cleanup
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SP = re.compile(r"[ \t ]+")


@dataclass
class Chunk:
    text: str
    article_number: str          # e.g. "254", "5 bis", "1", "preamble"
    chunk_index: int             # 0-based, within an article
    is_article: bool             # False for preamble / fallback chunks


def _normalize_num(raw: str) -> str:
    n = raw.strip().lower()
    n = n.replace("première", "1").replace("premier", "1")
    n = n.replace("1ère", "1").replace("1er", "1")
    return n


def _tidy(text: str) -> str:
    text = _MULTI_SP.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


def _subchunk(article_text: str, max_chars: int, min_chars: int) -> list[str]:
    """Split a long article on paragraph (blank-line) boundaries.

    Greedily accumulate paragraphs until ~max_chars, then emit. If a single
    paragraph itself exceeds max_chars, fall back to sentence splitting.
    """
    if len(article_text) <= max_chars:
        return [article_text] if article_text.strip() else []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", article_text) if p.strip()]
    out: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            # Flush current buffer first.
            if buf:
                out.append(buf.strip())
                buf = ""
            out.extend(_split_on_sentences(para, max_chars, min_chars))
            continue
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) <= max_chars or not buf:
            buf = candidate
        else:
            out.append(buf.strip())
            buf = para
    if buf.strip():
        out.append(buf.strip())
    # Merge any tiny trailing fragment into the previous chunk.
    if len(out) >= 2 and len(out[-1]) < min_chars:
        out[-2] = (out[-2] + "\n\n" + out[-1]).strip()
        out.pop()
    return [c for c in out if c.strip()]


def _split_on_sentences(text: str, max_chars: int, min_chars: int) -> list[str]:
    sents = re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÀ-Ý])", text)
    out, buf = [], ""
    for s in sents:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) <= max_chars or not buf:
            buf = cand
        else:
            out.append(buf.strip())
            buf = s
    if buf.strip():
        out.append(buf.strip())
    if len(out) >= 2 and len(out[-1]) < min_chars:
        out[-2] = (out[-2] + " " + out[-1]).strip()
        out.pop()
    return out


def chunk_by_articles(
    text: str,
    max_chars: int = 1400,
    min_chars: int = 200,
    min_articles: int = 2,
) -> list[Chunk]:
    """Split document text into per-article Chunks.

    A preamble chunk (article_number="preamble") is always emitted first if there
    is text before the first article, so introductions/definitions stay searchable.
    """
    text = _tidy(text)
    if not text:
        return []

    matches = list(ARTICLE_HEADER_RE.finditer(text))

    # --- Fallback: not article-structured -> paragraph chunking. ---
    if len(matches) < min_articles:
        return _fallback_chunks(text, max_chars, min_chars)

    chunks: list[Chunk] = []

    # Preamble (anything before the first article header).
    first_start = matches[0].start()
    # Strip a leading newline the regex consumes via (?:^|\n) -> trim it.
    pre = text[:first_start].lstrip("\n").strip()
    if pre:
        for piece in _subchunk(pre, max_chars, min_chars):
            chunks.append(Chunk(text=piece, article_number="preamble",
                                chunk_index=0, is_article=False))

    # Articles.
    for i, m in enumerate(matches):
        raw_label = m.group(1)
        raw_num = m.group(2)
        article_num = _normalize_num(raw_num)
        start = m.start()
        # If the match absorbed a leading "\n", keep the header line clean.
        header_lead = m.group(0)
        # Body runs until the next header (or end of text).
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        pieces = _subchunk(body, max_chars, min_chars)
        for j, piece in enumerate(pieces):
            chunks.append(Chunk(text=piece, article_number=article_num,
                                chunk_index=j, is_article=True))
    return chunks


def _fallback_chunks(text: str, max_chars: int, min_chars: int) -> list[Chunk]:
    """Paragraph-based chunking for documents without article numbering."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []
    out: list[Chunk] = []
    buf = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if buf:
                out.append(Chunk(buf, "preamble", 0, False))
                buf = ""
            for piece in _split_on_sentences(para, max_chars, min_chars):
                out.append(Chunk(piece, "preamble", 0, False))
            continue
        cand = (buf + "\n\n" + para).strip() if buf else para
        if len(cand) <= max_chars or not buf:
            buf = cand
        else:
            out.append(Chunk(buf, "preamble", 0, False))
            buf = para
    if buf.strip():
        out.append(Chunk(buf, "preamble", 0, False))
    if len(out) >= 2 and len(out[-1].text) < min_chars:
        out[-2] = Chunk(out[-2].text + "\n\n" + out[-1].text, "preamble", 0, False)
        out.pop()
    return out

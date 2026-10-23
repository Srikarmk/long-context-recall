"""Documents of a given length with facts placed at given depths.

An item's filler is a stream of WikiText paragraphs starting at its own article. The document for a
length is the stream cut at the last sentence end within that length's character budget, so the
document of every shorter length is a prefix of every longer one (the same text, only more of it).

A fact is placed at depth d (0 = start, 1 = end) at the sentence boundary nearest to d x the document's
length in characters, as a paragraph of its own; if that boundary falls inside a paragraph, the
paragraph is split there. The depth actually reached is recorded with every call.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import config as C
from .data import units

PARA = "\n\n"
# a sentence ends at . ! or ? (optionally followed by a closing quote or bracket) before a space and a
# capital letter, a digit or an opening quote; not after a known abbreviation or a single initial
SENT_END = re.compile(r"[.!?][\"')\]]?(?= +[A-Z0-9\"(])")
ABBREV = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "no", "nos", "co", "lt", "col", "gen", "capt", "sgt", "maj",
          "rev", "vs", "inc", "ltd", "mt", "ft", "fig", "vol", "pp", "op", "ca", "c", "approx", "dept", "est",
          "u.s", "u.k", "a.m", "p.m", "e.g", "i.e", "etc", "al", "jan", "feb", "mar", "apr", "jun", "jul", "aug",
          "sep", "sept", "oct", "nov", "dec", "hms", "sq", "km", "mi"}


@lru_cache(maxsize=256)
def stream(start: int, n_chars: int) -> str:
    """Paragraphs from unit `start` on (wrapping round the corpus) until at least n_chars characters."""
    u = units()
    out, total, k = [], 0, start
    while total < n_chars:
        out.append(u[k % len(u)])
        total += len(out[-1]) + len(PARA)
        k += 1
    return PARA.join(out)


def boundaries(text: str) -> list[int]:
    """Offsets where a fact may go: the start, the start of every paragraph, after every sentence end
    (pointing at the next sentence), and the end."""
    out = {0, len(text)}
    for m in re.finditer(re.escape(PARA), text):
        out.add(m.end())
    for m in SENT_END.finditer(text):
        word = re.search(r"([A-Za-z.]+)$", text[max(0, m.start() - 12):m.start()])
        w = word.group(1).lower().strip(".") if word else ""
        if w in ABBREV or (len(w) == 1 and w.isalpha()):
            continue
        j = m.end()
        while j < len(text) and text[j] == " ":
            j += 1
        out.add(j)
    return sorted(out)


def sentence_ends(text: str) -> list[int]:
    """Offsets just after a sentence or paragraph end: where a document may be cut."""
    out = []
    for b in boundaries(text):
        if b == 0:
            continue
        e = b
        while e > 0 and text[e - 1] in " \n":
            e -= 1
        out.append(e)
    return sorted(set(out))


@lru_cache(maxsize=1024)
def document(start: int, length: str) -> str:
    budget = C.doc_chars(length)
    text = stream(start, budget + 4000)
    cut = max(e for e in sentence_ends(text) if e <= budget)
    return text[:cut]


def place(doc: str, inserts: list[tuple[float, str]]) -> tuple[str, list[float]]:
    """Insert each (depth, paragraph) at the boundary nearest depth x len(doc). Inserts that land on the
    same boundary keep their given order. Returns the text and the depth reached by each insert."""
    bounds = boundaries(doc)
    n = len(doc)
    at = []
    for j, (d, text) in enumerate(inserts):
        target = d * n
        b = min(bounds, key=lambda x: (abs(x - target), x))
        at.append((b, j, text))
    pieces, last = [], 0
    for b, _, text in sorted(at):
        if b > last:
            pieces.append(doc[last:b].rstrip(" \n"))
        pieces.append(text)
        last = b
    if last < n:
        pieces.append(doc[last:].lstrip(" \n"))
    reached = [0.0] * len(inserts)
    for b, j, _ in at:
        reached[j] = round(b / n, 4) if n else 0.0
    return PARA.join(p for p in pieces if p), reached

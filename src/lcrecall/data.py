"""Download the pinned WikiText file and turn it into filler paragraphs.

WikiText-2 (raw) keeps Wikipedia's article and section structure as ` = Title = ` lines and spaces out
punctuation (` , `, ` @-@ `). The filler is the article prose only: section headings, lists and table
rows are dropped, punctuation is put back, and each article's title is kept as a line above its first
paragraph so a document reads as several encyclopedia articles one after another.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from functools import lru_cache

import pandas as pd

from . import config as C

TITLE_RE = re.compile(r"^ = ([^=].*?) = $")
HEAD_RE = re.compile(r"^ = = ")
DETOK = [(" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" , ", ", "), (" . ", ". "), (" ; ", "; "),
         (" : ", ": "), (" ! ", "! "), (" ? ", "? "), (" 's ", "'s "), (" n't ", "n't "), (" 're ", "'re "),
         (" 've ", "'ve "), (" 'll ", "'ll "), (" 'd ", "'d "), (" 'm ", "'m "), ("( ", "("), (" )", ")"),
         ("[ ", "["), (" ]", "]"), (" %", "%"), ("$ ", "$")]
MIN_WORDS = 40          # paragraphs shorter than this (captions, list items, table rows) are dropped
MIN_PARAS = 3           # articles with fewer kept paragraphs are dropped
MAX_DENSE = 0.06        # paragraphs where digits and non-ASCII characters exceed this share are dropped
                        # (score lines, coordinates, non-Latin names): they read badly and tokenize unevenly


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_path(name: str) -> str:
    return os.path.join(C.RAW, f"{name}.parquet")


def fetch() -> dict:
    os.makedirs(C.RAW, exist_ok=True)
    out = {}
    for name, (repo, rev, path, pinned) in C.SOURCES.items():
        dest = raw_path(name)
        if not os.path.exists(dest):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
            with urllib.request.urlopen(url, timeout=120) as r, open(dest + ".part", "wb") as f:
                f.write(r.read())
            os.replace(dest + ".part", dest)
        got = sha256(dest)
        if pinned and got != pinned:
            raise RuntimeError(f"{dest}: sha256 {got} does not match pinned {pinned}")
        out[name] = {"repo": repo, "revision": rev, "path": path, "bytes": os.path.getsize(dest), "sha256": got}
    with open(os.path.join(C.DATA, "sources.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def detok(s: str) -> str:
    s = " " + " ".join(s.split()) + " "
    for a, b in DETOK:
        s = s.replace(a, b)
    s = re.sub(r" ([.,;:!?])$", r"\1", s.strip())
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    return s


def prose(line: str) -> bool:
    words = line.split()
    dense = sum(c.isdigit() or ord(c) > 127 for c in line) / max(len(line), 1)
    return len(words) >= MIN_WORDS and line.count(" . ") + line.strip().endswith(".") >= 2 and dense <= MAX_DENSE


@lru_cache(maxsize=1)
def lines() -> tuple[str, ...]:
    return tuple(pd.read_parquet(raw_path("wikitext2_train"))["text"].tolist())


@lru_cache(maxsize=1)
def articles() -> tuple[dict, ...]:
    """Articles with at least MIN_PARAS prose paragraphs, in corpus order."""
    arts, cur = [], None
    for raw in lines():
        line = raw.rstrip("\n")
        m = TITLE_RE.match(line)
        if m and not HEAD_RE.match(line):
            if cur and len(cur["paras"]) >= MIN_PARAS:
                arts.append(cur)
            cur = {"title": detok(m.group(1)), "paras": []}
            continue
        if cur is None or HEAD_RE.match(line) or not line.strip():
            continue
        if prose(line):
            cur["paras"].append(detok(line))
    if cur and len(cur["paras"]) >= MIN_PARAS:
        arts.append(cur)
    return tuple(arts)


@lru_cache(maxsize=1)
def units() -> tuple[str, ...]:
    """The filler as a flat sequence of paragraphs; an article's first paragraph carries its title line."""
    out = []
    for a in articles():
        for j, p in enumerate(a["paras"]):
            out.append(f"{a['title']}\n{p}" if j == 0 else p)
    return tuple(out)


@lru_cache(maxsize=1)
def article_starts() -> tuple[int, ...]:
    starts, k = [], 0
    for a in articles():
        starts.append(k)
        k += len(a["paras"])
    return tuple(starts)


@lru_cache(maxsize=1)
def vocabulary() -> frozenset:
    """Every alphabetic word in the raw corpus, lower-cased: invented names must avoid all of them."""
    words = set()
    for line in lines():
        words.update(w.lower() for w in re.findall(r"[A-Za-z]+", line))
    return frozenset(words)


def corpus_facts() -> dict:
    u = units()
    return {"articles": len(articles()), "paragraphs": len(u), "chars": sum(len(x) for x in u),
            "words": sum(len(x.split()) for x in u)}

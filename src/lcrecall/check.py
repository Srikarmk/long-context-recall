"""The answer checker: which of the candidate answers an output names, by normalised word match.

Every candidate answer is an invented surname or town that occurs nowhere in the filler, so a match on
that one word is unambiguous. Outcomes, in order of precedence:

  correct       names the right answer and no near-miss answer
  mixed         names the right answer and a near-miss answer
  wrong_entity  names a near-miss answer (a distractor's person, or a decoy town) and not the right one
  bridge        two-hop only: names the intermediate person but no town
  abstain       names no candidate and declines ("not stated", "unknown", "the document does not say")
  other         anything else (another name, an empty or unreadable reply)

For a question whose fact is absent from the document the right reply is to abstain.
"""

from __future__ import annotations

import re
import unicodedata

ABSTAIN_RE = re.compile(
    r"\b(not stated|not (?:mentioned|given|provided|specified|included|found|listed|named|known|available)"
    r"|not in the (?:document|text)|(?:does not|doesn t|did not) (?:say|state|mention|give|specify|provide|name)"
    r"|no (?:information|mention|answer|name)|unknown|cannot (?:be )?determined?|can t (?:be )?determined?"
    r"|i (?:do not|don t) know|unable to)\b")
OUTCOMES = ["correct", "mixed", "wrong_entity", "bridge", "abstain", "other"]


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def has(t: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(norm(word))}\b", t) is not None


def classify(text: str, gold: list[str] | None, wrong: dict[str, list[str]], bridge: list[str] | None = None) -> dict:
    t = norm(text)
    g = gold is not None and any(has(t, w) for w in gold)
    hits = [lab for lab, ws in wrong.items() if any(has(t, w) for w in ws)]
    b = any(has(t, w) for w in bridge or [])
    abstained = t == "" or ABSTAIN_RE.search(t) is not None
    if g and not hits:
        out = "correct"
    elif g:
        out = "mixed"
    elif hits:
        out = "wrong_entity"
    elif b:
        out = "bridge"
    elif abstained and t:
        out = "abstain"
    else:
        out = "other"
    ok = out == "abstain" if gold is None else out == "correct"
    return {"outcome": out, "correct": ok, "picked": ",".join(hits), "answer_norm": t[:200]}

"""Build the item set deterministically: one fictional entity per item, with its distractors, its
second-hop fact and decoys, the start of its filler, and its A/A length. Pilot items come first."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from . import config as C
from .data import article_starts, units, vocabulary
from .facts import HOP2, HOP2_Q, RELATIONS, TAILS, NamePool

TYPES = [(rel, t) for rel, r in RELATIONS.items() for t in r["types"]]
AA_LENGTHS = ["1k", "4k", "8k"]
# Distractor depths are drawn inside these bands, one band each, so they never sit on a key position
# (0, 25, 50, 75, 100%) and are spread through the document.
DISTRACTOR_BANDS = [(0.05, 0.19), (0.31, 0.44), (0.56, 0.69), (0.81, 0.95)]
TWOHOP_DECOY_DEPTHS = [0.3, 0.7]


def entity_text(rel: str, typ: str, name: str, intro_ix: int, year: int, person: str) -> str:
    """One paragraph: an intro sentence about the entity, then the fact tying it to the person."""
    t = RELATIONS[rel]["types"][typ]
    return t["intro"][intro_ix].format(name=name) + " " + t["fact"].format(year=year, person=person)


def build_items(n_pilot: int = C.N_PILOT, n_main: int = C.N_MAIN, seed: int = C.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = NamePool(vocabulary(), seed)
    n = n_pilot + n_main
    order = [TYPES[i] for i in rng.permutation(len(TYPES))]
    types = [order[k % len(order)] for k in range(n)]
    starts = article_starts()
    art = rng.permutation(len(starts))[:n]
    items = []
    for k in range(n):
        rel, typ = types[k]
        r = RELATIONS[rel]
        stem, near_a, near_b = pool.stem_family()
        other_types = [t for t in r["types"] if t != typ]
        alt_type = other_types[int(rng.integers(len(other_types)))]
        people = pool.people(6)          # key, three distractors, two decoys
        years = rng.choice(np.arange(1820, 1936), size=4, replace=False)
        intros = rng.permutation(3)      # key and its two same-type distractors use different intros
        name = f"{stem} {typ}"
        key_person = people[0]
        needle = entity_text(rel, typ, name, int(intros[0]), int(years[0]), key_person["name"])
        dspec = [("same_stem", f"{stem} {alt_type}", alt_type, int(rng.integers(3))),
                 ("same_prefix", f"{near_a} {typ}", typ, int(intros[1])),
                 ("same_suffix", f"{near_b} {typ}", typ, int(intros[2]))]
        distractors = []
        for j, (kind, dname, dtype, ix) in enumerate(dspec):
            p = people[1 + j]
            distractors.append({"kind": kind, "name": dname, "person": p,
                                "text": entity_text(rel, dtype, dname, ix, int(years[1 + j]), p["name"])})
        bands = sorted(rng.choice(len(DISTRACTOR_BANDS), size=3, replace=False))
        ddepths = [round(float(rng.uniform(*DISTRACTOR_BANDS[b])), 3) for b in bands]
        order_d = rng.permutation(3)     # which distractor goes in which band
        for j, d in enumerate(distractors):
            d["depth"] = ddepths[int(order_d[j])]
        towns = [pool.town() for _ in range(3)]
        tails = rng.choice(len(TAILS), size=3, replace=False)
        hop2 = {"town": towns[0], "text": HOP2.format(person=key_person["name"], town=towns[0], tail=TAILS[tails[0]])}
        decoys = [{"person": people[4 + j], "town": towns[1 + j], "depth": TWOHOP_DECOY_DEPTHS[j],
                   "text": HOP2.format(person=people[4 + j]["name"], town=towns[1 + j], tail=TAILS[tails[1 + j]])}
                  for j in range(2)]
        split = "pilot" if k < n_pilot else "main"
        items.append({
            "id": f"{split}-{k if split == 'pilot' else k - n_pilot:02d}", "split": split, "relation": rel,
            "type": typ, "name": name, "stem": stem,
            "question": r["question"].format(name=name),
            "twohop_question": r["twohop"].format(name=name),
            "hop2_question": HOP2_Q.format(person=key_person["name"]),
            "answer": key_person, "year": int(years[0]), "needle": needle,
            "distractors": distractors, "hop2": hop2, "decoys": decoys,
            "filler_start": int(starts[int(art[k])]),
            "aa_length": AA_LENGTHS[(k - n_pilot) % len(AA_LENGTHS)] if split == "main" else "4k",
        })
    return items


def check_items(items: list[dict]) -> None:
    """Invariants the analysis relies on; raises on the first violation."""
    vocab = vocabulary()
    seen = Counter()
    for it in items:
        people = [it["answer"]] + [d["person"] for d in it["distractors"]] + [d["person"] for d in it["decoys"]]
        lasts = [p["last"].lower() for p in people]
        firsts = [p["first"].lower() for p in people]
        assert len(set(lasts)) == len(lasts) and len(set(firsts)) == len(firsts), it["id"]
        towns = [it["hop2"]["town"]] + [d["town"] for d in it["decoys"]]
        stems = [it["stem"]] + [d["name"].split()[0] for d in it["distractors"][1:]]
        for w in lasts + [t.lower() for t in towns] + [s.lower() for s in stems]:
            assert w not in vocab, (it["id"], w)
        names = [it["name"]] + [d["name"] for d in it["distractors"]]
        assert len(set(names)) == len(names), it["id"]
        assert it["answer"]["name"] in it["needle"] and it["name"] in it["needle"]
        for d in it["distractors"]:
            assert d["person"]["name"] in d["text"] and d["name"] in d["text"]
            assert it["answer"]["last"] not in d["text"]
        seen.update(lasts + [t.lower() for t in towns])
    dup = [w for w, c in seen.items() if c > 1]
    assert not dup, f"names reused across items: {dup[:5]}"
    assert len(units()) > 1000


def write_items() -> list[dict]:
    items = build_items()
    check_items(items)
    with open(C.ITEMS, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


def load_items(split: str | None = "main") -> list[dict]:
    with open(C.ITEMS) as f:
        items = [json.loads(line) for line in f]
    return [it for it in items if split is None or it["split"] == split]

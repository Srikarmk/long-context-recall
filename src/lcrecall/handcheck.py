"""Check of the answer checker against a hand-labelled sample.

`sample` writes handcheck/sample.csv: a seeded random sample of main-run outputs, stratified by task,
plus outputs the checker put in `other` or `mixed` or that are longer than a short answer, where
mistakes are likelier. Each row has the question, the right answer and the near-miss answers it could be
confused with, and the raw output, but not the checker's reading, which is kept apart in
handcheck/parser_key.csv. The `label` column is left empty for a human labeller (Srikar) to fill with one
of: correct, mixed, wrong_entity, bridge, abstain, other (definitions in check.py and the README).
Saved as handcheck/labels.csv, `score` compares the labels with the checker and writes
handcheck/agreement.json.
"""

from __future__ import annotations

import json
import os

import pandas as pd

from . import config as C
from .analysis import load_rows
from .items import load_items
from .prompts import Cell

COLS = ["id", "subset", "model", "item", "cell", "question", "right_answer", "near_miss_answers", "output"]
SHARE = {"key": 0.35, "distract": 0.25, "absent": 0.15, "twohop": 0.2, "closed": 0.05}


def _group(r) -> str:
    if r.task == "key":
        return "distract" if r.variant == "distract" else "key"
    return "twohop" if r.task in ("twohop", "hop2") else r.task


def _question_and_answers(it: dict, cell: Cell) -> tuple[str, str, str]:
    if cell.task in ("twohop", "hop2") or (cell.task == "closed" and cell.variant == "hop2"):
        q = it["twohop_question"] if cell.task == "twohop" else it["hop2_question"]
        near = [d["town"] for d in it["decoys"]] + ([it["answer"]["name"] + " (the person, not a town)"]
                                                    if cell.task == "twohop" else [])
        return q, it["hop2"]["town"], "; ".join(near)
    right = "(none: the fact is not in the document)" if cell.task == "absent" else it["answer"]["name"]
    return it["question"], right, "; ".join(d["person"]["name"] for d in it["distractors"])


def sample(n: int = 60, n_hard: int = 20, seed: int = 11, split: str = "main") -> None:
    os.makedirs(C.HANDCHECK, exist_ok=True)
    df = load_rows(split)
    if df.empty:
        raise SystemExit("no results yet")
    pre = "" if split == "main" else f"{split}_"
    df = df.assign(group=[_group(r) for r in df.itertuples()]).reset_index(drop=True)
    parts = []
    for g, share in SHARE.items():
        d = df[df.group == g]
        k = min(len(d), int(round(n * share)))
        if k:
            parts.append(d.sample(n=k, random_state=seed))
    rand = pd.concat(parts).sample(frac=1, random_state=seed).assign(subset="random")
    rest = df.drop(rand.index)
    hard = rest[rest.outcome.isin(["other", "mixed"]) | (rest.output.str.len() > 60)]
    hard = hard.sample(n=min(n_hard, len(hard)), random_state=seed).assign(subset="hard")
    s = pd.concat([rand, hard]).reset_index(drop=True)
    items = {it["id"]: it for it in load_items(split)}
    qa = [_question_and_answers(items[r.id], Cell.from_key(r.cell)) for r in s.itertuples()]
    s["question"], s["right_answer"], s["near_miss_answers"] = zip(*qa) if qa else ([], [], [])
    s = s.rename(columns={"id": "item"})
    s["id"] = range(1, len(s) + 1)
    s[COLS].assign(label="").to_csv(os.path.join(C.HANDCHECK, f"{pre}sample.csv"), index=False)
    # the checker's reading, kept apart so the labeller does not see it
    s[["id", "outcome"]].to_csv(os.path.join(C.HANDCHECK, f"{pre}parser_key.csv"), index=False)
    print(f"wrote {len(s)} rows to handcheck/{pre}sample.csv ({len(rand)} random, {len(hard)} hard); labels empty")


def score(split: str = "main") -> dict:
    pre = "" if split == "main" else f"{split}_"
    path = os.path.join(C.HANDCHECK, f"{pre}labels.csv")
    if not os.path.exists(path):
        raise SystemExit(f"handcheck/{pre}labels.csv not found: copy {pre}sample.csv to {pre}labels.csv and "
                         "fill the label column")
    lab = pd.read_csv(path, keep_default_na=False)
    lab = lab[lab.label.astype(str).str.strip() != ""]
    if lab.empty:
        raise SystemExit("no labels filled in yet")
    key = pd.read_csv(os.path.join(C.HANDCHECK, f"{pre}parser_key.csv"), keep_default_na=False)
    lab = lab.merge(key, on="id")
    lab["agree"] = lab.label.str.strip().str.lower() == lab.outcome
    out = {"n": int(len(lab)), "agree": int(lab.agree.sum()), "by_subset": {}}
    for sub, g in lab.groupby("subset", sort=False):
        out["by_subset"][sub] = {"n": int(len(g)), "agree": int(g.agree.sum())}
    out["disagreements"] = [{"id": int(r.id), "subset": r.subset, "label": r.label, "checker": r.outcome,
                             "output": r.output[-300:]} for r in lab[~lab.agree].itertuples()]
    parts = [f"{sub}: {v['agree']}/{v['n']}" for sub, v in out["by_subset"].items()]
    out["summary"] = ("A labelling pass by Srikar read each sampled output without seeing the checker's reading "
                      "and assigned an outcome. Agreement with the checker: " + "; ".join(parts) + ".")
    with open(os.path.join(C.HANDCHECK, f"{pre}agreement.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(out["summary"])
    return out

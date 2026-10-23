"""Tables computed from results/rows/: accuracy by length and depth, the effects of length, position and
distractors, fact-absent replies, two-hop composition, the A/A floor, the closed-book control and run
facts. Every comparison between cells is paired by item."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .check import OUTCOMES
from .items import load_items
from .prompts import TWOHOP_PLACEMENTS, Cell
from .runner import read_rows, score_row
from .stats import item_ci, mean_ci, paired_ci

LEN_ORDER = list(C.LENGTHS)
SHORT = ["1k", "4k", "8k"]


def load_rows(split: str = "main") -> pd.DataFrame:
    rows = []
    for m in C.MODELS:
        rows += read_rows(m, split)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # re-check every stored output with the current checker, so all rows are scored by the same rules
    items = {it["id"]: it for it in load_items(split)}
    checked = [score_row(items[r["id"]], Cell.from_key(r["cell"]), r["output"]) for r in df.to_dict("records")]
    for k in checked[0]:
        df[k] = [c[k] for c in checked]
    df["correct"] = df["correct"].astype(float)
    return df


def models_in(df: pd.DataFrame) -> list[str]:
    return [m for m in C.MODELS if not df.empty and m in set(df.model)]


def mlabel(m: str) -> str:
    return f"{m} (thinking)" if m in C.THINKING_ONLY else m


def key_rows(df: pd.DataFrame, model: str, variant: str = "plain") -> pd.DataFrame:
    """Key-fact cells (no repeat tag) without (`plain`) or with (`distract`) distractors."""
    return df[(df.model == model) & (df.task == "key") & (df.variant == variant) & (df.tag == "")]


def grid(df: pd.DataFrame, model: str, variant: str = "plain") -> dict:
    """(length, depth) -> accuracy over items; (length, None) -> accuracy over all depths."""
    d = key_rows(df, model, variant)
    out = {}
    for L in LEN_ORDER:
        dl = d[d.length == L]
        if dl.empty:
            continue
        for dep in C.DEPTHS:
            out[(L, dep)] = item_ci(dl[dl.depth == dep], "correct")
        out[(L, None)] = item_ci(complete(dl), "correct")
    return out


def complete(d: pd.DataFrame) -> pd.DataFrame:
    """Rows of the items answered at every depth (so a length's mean is over the same depths per item)."""
    n = d.groupby("id").depth.nunique()
    return d[d.id.isin(n[n == len(C.DEPTHS)].index)]


def length_effect(df: pd.DataFrame, model: str, variant: str = "plain", ref: str = "1k") -> dict:
    """Accuracy at each length minus at `ref`, all depths, paired by item."""
    d = key_rows(df, model, variant)
    base = complete(d[d.length == ref])
    return {L: paired_ci(complete(d[d.length == L]), "correct", base) for L in LEN_ORDER
            if L != ref and not d[d.length == L].empty}


def _by_item_depth(d: pd.DataFrame) -> pd.DataFrame:
    return d.pivot_table(index="id", columns="depth", values="correct", aggfunc="mean")


def position_effect(df: pd.DataFrame, model: str, lengths: list[str] | None = None, variant: str = "plain") -> dict:
    """Middle minus ends: acc(50%) - (acc(0) + acc(100%)) / 2, and start minus end, and the interior
    (25/50/75%) minus the ends, paired by item. Pooled over `lengths` by averaging an item's lengths."""
    d = key_rows(df, model, variant)
    if lengths:
        d = d[d.length.isin(lengths)]
    if d.empty:
        return {}
    p = _by_item_depth(complete_by_length(d)).dropna()
    if p.empty:
        return {}
    ends = (p[0.0] + p[1.0]) / 2
    return {"middle_minus_ends": mean_ci((p[0.5] - ends).values),
            "interior_minus_ends": mean_ci((p[[0.25, 0.5, 0.75]].mean(axis=1) - ends).values),
            "start_minus_end": mean_ci((p[0.0] - p[1.0]).values),
            "worst": float(p.mean().min()), "best": float(p.mean().max()),
            "worst_depth": float(p.mean().idxmin()), "n": int(len(p))}


def complete_by_length(d: pd.DataFrame) -> pd.DataFrame:
    """Keep (item, length) groups answered at every depth."""
    n = d.groupby(["id", "length"]).depth.nunique()
    keep = n[n == len(C.DEPTHS)].index
    return d.set_index(["id", "length"]).loc[lambda x: x.index.isin(keep)].reset_index()


def outcome_shares(d: pd.DataFrame) -> dict:
    """Share of each outcome over items (an item's calls averaged first)."""
    out = {}
    for o in OUTCOMES:
        out[o] = item_ci(d.assign(_o=(d.outcome == o).astype(float)), "_o") if not d.empty else mean_ci([])
    return out


def distractor_tables(df: pd.DataFrame, model: str, length: str = C.BASE_LENGTH) -> dict:
    m = df[(df.model == model) & (df.length == length) & (df.tag == "")]
    plain = m[(m.task == "key") & (m.variant == "plain")]
    dist = m[(m.task == "key") & (m.variant == "distract")]
    if dist.empty:
        return {}
    out = {"n_items": int(dist.id.nunique()), "acc_plain": item_ci(plain[plain.id.isin(dist.id)], "correct"),
           "acc_distract": item_ci(dist, "correct"), "effect": paired_ci(dist, "correct", plain),
           "by_depth": {}, "shares": outcome_shares(dist), "picked": {}}
    for dep in C.DEPTHS:
        a, b = dist[dist.depth == dep], plain[plain.depth == dep]
        out["by_depth"][dep] = {"plain": item_ci(b[b.id.isin(a.id)], "correct"), "distract": item_ci(a, "correct"),
                                "effect": paired_ci(a, "correct", b)}
    wrong = dist[dist.outcome.isin(["wrong_entity", "mixed"])]
    for kind in ("same_stem", "same_prefix", "same_suffix"):
        out["picked"][kind] = int(wrong.picked.str.contains(kind).sum())
    out["n_calls"] = int(len(dist))
    ab = m[m.task == "absent"]
    out["absent"] = {v: {"n": int(ab[ab.variant == v].id.nunique()), "shares": outcome_shares(ab[ab.variant == v])}
                     for v in ("plain", "distract") if not ab[ab.variant == v].empty}
    return out


def picked_depth(df: pd.DataFrame, model: str) -> dict:
    """Of wrong-entity replies with distractors (one distractor named), how often the named distractor sits
    before the key fact, against the chance level of naming one of the three at random (the share of the
    three that sit before the key fact in that call), and how often it is the one nearest the key fact."""
    d = df[(df.model == model) & (df.task == "key") & (df.variant == "distract") & (df.tag == "")
           & (df.outcome == "wrong_entity")]
    kinds3 = ("same_stem", "same_prefix", "same_suffix")
    before, chance, nearest, n = 0, 0.0, 0, 0
    for r in d.itertuples():
        reached = r.reached or {}
        kinds = [k for k in r.picked.split(",") if k in reached]
        if len(kinds) != 1 or "key" not in reached or not all(k in reached for k in kinds3):
            continue
        k, key = kinds[0], reached["key"]
        n += 1
        before += reached[k] < key
        chance += sum(reached[x] < key for x in kinds3) / 3
        dist = {x: abs(reached[x] - key) for x in kinds3}
        nearest += min(dist, key=dist.get) == k
    if not n:
        return {}
    return {"n": n, "before": before, "chance_before": chance / n, "nearest": nearest}


def twohop_tables(df: pd.DataFrame, model: str, length: str = C.BASE_LENGTH) -> dict:
    m = df[(df.model == model) & (df.length == length) & (df.tag == "")]
    th = m[m.task == "twohop"]
    if th.empty:
        return {}
    out = {"n_items": int(th.id.nunique()), "acc": {}, "shares": {}, "vs_near": {}}
    for v in TWOHOP_PLACEMENTS:
        d = th[th.variant == v]
        out["acc"][v] = item_ci(d, "correct")
        out["shares"][v] = outcome_shares(d)
    near = th[th.variant == "near"]
    for v in ("far_fwd", "far_back"):
        out["vs_near"][v] = paired_ci(th[th.variant == v], "correct", near)
    hop2 = m[m.task == "hop2"]
    single = m[(m.task == "key") & (m.variant == "plain")]
    out["hop2"] = item_ci(hop2, "correct")
    out["single"] = item_ci(single[single.id.isin(th.id)], "correct")
    ff = th[th.variant == "far_fwd"]
    out["fwd_minus_hop2"] = paired_ci(ff, "correct", hop2)
    out["fwd_minus_single"] = paired_ci(ff, "correct", single)
    return out


def aa_tables(df: pd.DataFrame, model: str) -> dict:
    """The A/A floor: share of repeated calls whose correctness (flip) or answer text changed."""
    m = df[df.model == model]
    rep = m[m.tag == "aa"]
    if rep.empty:
        return {}
    orig = m[m.tag == ""].set_index(["id", "length", "depth", "task", "variant"])
    r = rep.set_index(["id", "length", "depth", "task", "variant"])
    o = orig[["correct", "answer_norm", "output"]].add_suffix("_o")
    j = r.join(o, how="inner").reset_index()
    j["flip"] = (j.correct != j.correct_o).astype(float)
    j["text_changed"] = (j.answer_norm != j.answer_norm_o).astype(float)
    j["raw_changed"] = (j.output != j.output_o).astype(float)
    out = {"n_calls": int(len(j)), "n_items": int(j.id.nunique()), "flip": item_ci(j, "flip"),
           "text_changed": item_ci(j, "text_changed"), "raw_changed": item_ci(j, "raw_changed"),
           "n_flips": int(j.flip.sum()), "by_length": {}}
    for L in LEN_ORDER:
        jl = j[j.length == L]
        if not jl.empty:
            out["by_length"][L] = {"n": int(len(jl)), "flips": int(jl.flip.sum()), "flip": item_ci(jl, "flip")}
    return out


def closed_tables(df: pd.DataFrame, model: str) -> dict:
    d = df[(df.model == model) & (df.task == "closed")]
    if d.empty:
        return {}
    return {v: {"n": int(len(d[d.variant == v])), "shares": outcome_shares(d[d.variant == v])}
            for v in ("key", "hop2") if not d[d.variant == v].empty}


def run_facts(df: pd.DataFrame) -> list[dict]:
    out = []
    for m in models_in(df):
        d = df[df.model == m]
        for L in ["closed"] + LEN_ORDER:
            x = d[d.task == "closed"] if L == "closed" else d[(d.length == L) & (d.task != "closed")]
            if x.empty:
                continue
            ratio = x.prompt_tokens / x.est_tokens
            out.append({"model": m, "length": L, "calls": int(len(x)), "items": int(x.id.nunique()),
                        "prompt_tokens": float(x.prompt_tokens.mean()), "max_tokens": int(x.prompt_tokens.max()),
                        "ratio_lo": float(ratio.min()), "ratio_hi": float(ratio.max()),
                        "seconds": float(x.seconds.median()), "wall": float(x.wall_seconds.median()),
                        "prefill_tps": float((x.prompt_tokens / x.prefill_seconds.clip(lower=1e-3)).median()),
                        "output_tokens": float(x.output_tokens.mean()), "truncated": float(x.truncated.mean()),
                        "thinking_chars": float(x.thinking_chars.mean()),
                        "shares": {o: float((x.outcome == o).mean()) for o in OUTCOMES}})
    return out


def pilot_rates() -> dict:
    """Mean wall seconds per call from the pilot rows: by (model, stage) and by (model, length)."""
    df = load_rows("pilot")
    rates = {}
    if df.empty:
        return rates
    # cached rows carry the wall time of the call that filled the cache, so they count too
    # drop stalls (a call more than 10 times its group's median: one server hang in the pilot)
    med = df.groupby(["model", "stage"]).wall_seconds.transform("median")
    df = df[df.wall_seconds <= 10 * med]
    for (m, st), g in df.groupby(["model", "stage"]):
        rates[(m, st)] = float(g.wall_seconds.mean())
    for (m, L), g in df[df.task != "closed"].groupby(["model", "length"]):
        rates[(m, L)] = float(g.wall_seconds.mean())
    for m, g in df[df.task == "closed"].groupby("model"):
        rates[(m, "closed")] = float(g.wall_seconds.mean())
    return rates


def floor_of(df: pd.DataFrame, model: str) -> float:
    a = aa_tables(df, model)
    return a["flip"]["mean"] if a else float("nan")


def counts(r: dict, floor: float) -> bool:
    """The floor rule: an effect counts when its interval excludes zero and its size exceeds the model's
    A/A flip rate."""
    if not r or not r.get("n"):
        return False
    return (r["lo"] > 0 or r["hi"] < 0) and abs(r["mean"]) > (0 if np.isnan(floor) else floor)

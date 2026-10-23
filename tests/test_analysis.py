import numpy as np
import pandas as pd

from lcrecall import analysis as A
from lcrecall import config as C


def rows(model="qwen3:8b", n=20, acc=None, tag="", variant="plain", lengths=("1k", "4k"), task="key"):
    """acc(length, depth, item) -> 0/1."""
    out = []
    for i in range(n):
        for L in lengths:
            for d in C.DEPTHS:
                c = float(acc(L, d, i))
                out.append({"model": model, "id": f"m{i:02d}", "task": task, "length": L, "depth": d,
                            "variant": variant, "tag": tag, "correct": c, "outcome": "correct" if c else "wrong_entity",
                            "answer_norm": "a" if c else "b", "output": "a" if c else "b", "picked": "" if c else "same_stem"})
    return pd.DataFrame(out)


def test_grid_and_length_effect():
    df = rows(acc=lambda L, d, i: L == "1k" or i % 2 == 0)
    g = A.grid(df, "qwen3:8b")
    assert g[("1k", None)]["mean"] == 1.0 and abs(g[("4k", None)]["mean"] - 0.5) < 1e-12
    le = A.length_effect(df, "qwen3:8b")
    assert abs(le["4k"]["mean"] + 0.5) < 1e-12 and le["4k"]["hi"] < 0


def test_position_effect_detects_middle_dip():
    df = rows(acc=lambda L, d, i: d in (0.0, 1.0) or (d != 0.5 and i % 2 == 0))
    p = A.position_effect(df, "qwen3:8b", ["1k", "4k"])
    assert p["middle_minus_ends"]["mean"] == -1.0 and p["worst_depth"] == 0.5
    assert abs(p["start_minus_end"]["mean"]) < 1e-12


def test_aa_flip_rate_and_floor_rule():
    a = rows(lengths=("1k",), acc=lambda L, d, i: 1)
    b = rows(lengths=("1k",), tag="aa", acc=lambda L, d, i: not (i == 0 and d == 0.0))
    t = A.aa_tables(pd.concat([a, b]), "qwen3:8b")
    assert t["n_flips"] == 1 and t["n_calls"] == 20 * len(C.DEPTHS)
    assert abs(t["flip"]["mean"] - 1 / (20 * 5)) < 1e-12
    assert A.counts({"mean": -0.3, "lo": -0.4, "hi": -0.2, "n": 20}, 0.01)
    assert not A.counts({"mean": -0.3, "lo": -0.4, "hi": 0.1, "n": 20}, 0.01)
    assert not A.counts({"mean": -0.005, "lo": -0.008, "hi": -0.002, "n": 20}, 0.01)


def test_distractor_effect_is_paired():
    L = C.BASE_LENGTH
    plain = rows(lengths=(L,), acc=lambda *_: 1)
    dist = rows(lengths=(L,), variant="distract", acc=lambda L_, d, i: d != 0.5)
    t = A.distractor_tables(pd.concat([plain, dist]), "qwen3:8b")
    assert abs(t["effect"]["mean"] + 0.2) < 1e-12
    assert t["by_depth"][0.5]["effect"]["mean"] == -1.0 and t["picked"]["same_stem"] == 20
    assert np.isclose(t["shares"]["wrong_entity"]["mean"], 0.2)

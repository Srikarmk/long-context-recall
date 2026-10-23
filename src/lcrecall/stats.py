"""Bootstrap statistics that resample items, never single calls.

Adapted from llm-judge-bias and prompt-brittleness (themselves from memory-rot and agent-interface-lab).
The unit is the item: the same fictional fact appears at every length and depth, so an item's rows move
together and every comparison between cells is paired by item.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def boot_means(values: np.ndarray, n_boot: int = C.BOOT, seed: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return values[idx].mean(axis=1)


def mean_ci(values, n_boot: int = C.BOOT, seed: int = 0) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    b = boot_means(values, n_boot, seed)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(values))}


def item_ci(df: pd.DataFrame, value: str, unit: str = "id", n_boot: int = C.BOOT, seed: int = 0) -> dict:
    """Mean of `value` over items (averaging an item's rows first), item bootstrap."""
    if df.empty:
        return mean_ci([])
    per = df.groupby(unit)[value].mean()
    return mean_ci(per.values, n_boot, seed)


def paired_ci(df: pd.DataFrame, value: str, ref: pd.DataFrame, ref_value: str | None = None, unit: str = "id",
              n_boot: int = C.BOOT, seed: int = 0) -> dict:
    """Mean over items of (item mean of `value` in df) minus (item mean in ref), on items in both."""
    if df.empty or ref.empty:
        return mean_ci([])
    a = df.groupby(unit)[value].mean()
    b = ref.groupby(unit)[ref_value or value].mean()
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    return mean_ci((m["a"] - m["b"]).values, n_boot, seed)


def above_zero(r: dict) -> bool:
    return bool(r and r.get("n") and r["lo"] > 0)


def below_zero(r: dict) -> bool:
    return bool(r and r.get("n") and r["hi"] < 0)


def excludes_zero(r: dict) -> bool:
    return above_zero(r) or below_zero(r)

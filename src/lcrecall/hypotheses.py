"""Scoring of the pre-registered predictions in HYPOTHESES.md against the computed tables."""

from __future__ import annotations

import numpy as np

from . import analysis as A
from . import config as C
from .stats import mean_ci, paired_ci


def _pct(r: dict) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{100 * r['mean']:.0f}% [{100 * r['lo']:.0f}, {100 * r['hi']:.0f}]"


def _pp(r: dict) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{100 * r['mean']:+.1f} pp [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]"


def _v(ok: bool | None) -> str:
    return "not testable" if ok is None else ("held" if ok else "did not hold")


def _below(r: dict, floor: float = 0.0) -> bool:
    return bool(r and r.get("n") and r["hi"] < 0 and -r["mean"] > (0 if np.isnan(floor) else floor))


def _share(d: dict, outcome: str) -> float:
    return d["shares"][outcome]["mean"] if d and d["shares"][outcome]["n"] else float("nan")


def hypothesis_lines(df, grids, dgrids, le, dle, dts, ths, aa, cb) -> list[str]:
    main = [m for m in C.MAIN_MODELS if m in grids and grids[m]]
    if not main:
        return ["No main-model results yet.\n"]
    floor = {m: A.floor_of(df, m) for m in grids}
    out = []

    # 1 closed book
    parts, ok, have = [], True, False
    for m in grids:
        for v, r in (cb.get(m) or {}).items():
            have = True
            c = r["shares"]["correct"]["mean"]
            ok &= c == 0
            parts.append(f"{m} {v} {100 * c:.0f}%")
    out.append(f"1. No item is answered correctly closed-book by any model: {_v(ok if have else None)} "
               f"({'; '.join(parts)}).")

    # 2 fact alone near ceiling through 16k
    parts, ok, have = [], True, 0
    for m in main:
        for L in ("1k", "4k", "8k", "16k"):
            r = grids[m].get((L, None))
            if r and r["n"]:
                have += 1
                ok &= r["mean"] >= 0.9
                parts.append(f"{m} {L} {100 * r['mean']:.0f}%")
    out.append(f"2. With the fact alone, recall over all depths is at least 90% at every length from 1k to 16k for every "
               f"main model: {_v(ok if have == 4 * len(main) else (False if not ok else None))} ({'; '.join(parts)}).")

    # 3 length alone matters little through 16k
    parts, ok, have = [], True, False
    for m in main:
        r = le[m].get("16k")
        if r and r["n"]:
            have = True
            ok &= r["lo"] >= -0.05
            parts.append(f"{m} {_pp(r)}")
    out.append(f"3. With the fact alone, 16k minus 1k is no lower than -5 pp at the lower end of its interval for every "
               f"main model: {_v(ok if have and len(parts) == len(main) else None)} ({'; '.join(parts)}).")

    # 4 distractors lower recall at 4k beyond the floor
    parts, n = [], 0
    for m in main:
        r = (dts[m].get(C.BASE_LENGTH) or {}).get("effect")
        n += _below(r, floor[m])
        parts.append(f"{m} {_pp(r)}")
    out.append(f"4. At {C.BASE_LENGTH}, near-miss distractors lower recall (interval below zero, larger than the A/A "
               f"flip rate) for every main model: {_v(n == len(main))} ({'; '.join(parts)}).")

    # 5 middle worse than ends with distractors, 1k-8k
    parts, n = [], 0
    for m in main:
        p = A.position_effect(df, m, A.SHORT, "distract")
        r = p.get("middle_minus_ends") if p else None
        n += _below(r, floor[m])
        parts.append(f"{m} {_pp(r)}")
    out.append(f"5. With distractors, pooled over 1k to 8k, recall with the fact in the middle is below the mean of start "
               f"and end (interval below zero, beyond the floor) for at least two main models: {_v(n >= 2)} "
               f"({'; '.join(parts)}).")

    # 6 distractor recall falls with length
    parts, n, have = [], 0, 0
    for m in main:
        r = dle[m].get("16k")
        if r and r["n"]:
            have += 1
        n += _below(r, floor[m])
        parts.append(f"{m} {_pp(r)}")
    out.append(f"6. With distractors, recall at 16k is below recall at 1k (interval below zero, beyond the floor) for at "
               f"least two main models: {_v(n >= 2 if have >= 2 else None)} ({'; '.join(parts)}).")

    # 7 wrong picks favour distractors placed before the key fact
    parts, n = [], 0
    for m in main:
        p = A.picked_depth(df, m)
        if p:
            share = p["before"] / p["n"]
            n += share > p["chance_before"] + 0.10
            parts.append(f"{m} {100 * share:.0f}% of {p['n']} against {100 * p['chance_before']:.0f}% by chance")
    out.append(f"7. When a reply names a distractor, it names one placed before the key fact more often than chance, by "
               f"more than 10 pp, for at least two main models: {_v(n >= 2 if len(parts) >= 2 else None)} "
               f"({'; '.join(parts)}).")

    # 8 fact absent
    parts, ok_plain, n_drop, have = [], True, 0, False
    for m in main:
        ab = (dts[m].get(C.BASE_LENGTH) or {}).get("absent", {})
        if "plain" in ab and "distract" in ab:
            have = True
            a0, a1 = _share(ab["plain"], "abstain"), _share(ab["distract"], "abstain")
            ok_plain &= a0 >= 0.9
            sub = df[(df.model == m) & (df.task == "absent") & (df.length == C.BASE_LENGTH) & (df.tag == "")]
            sub = sub.assign(_a=(sub.outcome == "abstain").astype(float))
            r = paired_ci(sub[sub.variant == "distract"], "_a", sub[sub.variant == "plain"])
            n_drop += _below(r)
            parts.append(f"{m} abstains {100 * a0:.0f}% alone, {100 * a1:.0f}% with distractors ({_pp(r)})")
    out.append(f"8. With the fact left out, every main model abstains on at least 90% of items when there are no "
               f"distractors, and abstains less with distractors (interval below zero) for at least two: "
               f"{_v(ok_plain and n_drop >= 2 if have else None)} ({'; '.join(parts)}).")

    # 9 two-hop: adjacent high, far lower for at least one, and far forward below the second hop alone
    t4 = {m: ths[m].get("4k") for m in main}
    if all(t4.values()):
        near_ok = all(t["acc"]["near"]["mean"] >= 0.9 for t in t4.values())
        far_drop = sum(_below(t["vs_near"]["far_fwd"], floor[m]) or _below(t["vs_near"]["far_back"], floor[m])
                       for m, t in t4.items())
        gap = sum(_below(t["fwd_minus_hop2"], floor[m]) for m, t in t4.items())
        out.append(f"9. Two-hop at 4k: adjacent facts answered at least 90% by every main model, at least one model "
                   f"lower with the facts far apart (either order, beyond the floor), and at least one lower on the "
                   f"far two-hop question than on its second hop alone: {_v(near_ok and far_drop >= 1 and gap >= 1)} ("
                   + "; ".join(f"{m} adjacent {_pct(t['acc']['near'])}, far {_pct(t['acc']['far_fwd'])} / "
                               f"{_pct(t['acc']['far_back'])}, second hop alone {_pct(t['hop2'])}" for m, t in t4.items())
                   + ").")
    else:
        out.append("9. Two-hop at 4k: not testable (not run for every main model).")

    # 10 two-hop worse at 16k
    parts, n, have = [], 0, 0
    for m in main:
        a, b = ths[m].get("4k"), ths[m].get("16k")
        if not (a and b):
            continue
        have += 1
        sub = df[(df.model == m) & (df.task == "twohop") & (df.tag == "") & (df.variant != "near")]
        r = paired_ci(sub[sub.length == "16k"], "correct", sub[sub.length == "4k"])
        n += _below(r, floor[m])
        parts.append(f"{m} {_pp(r)}")
    out.append(f"10. Two-hop with the facts far apart (both orders) is less accurate at 16k than at 4k (interval below "
               f"zero, beyond the floor) for at least two main models: {_v(n >= 2 if have >= 2 else None)} "
               f"({'; '.join(parts)}).")

    # 11 A/A floor
    parts, ok, have = [], True, False
    for m in grids:
        a = aa.get(m)
        if a:
            have = True
            ok &= a["flip"]["mean"] < 0.03
            parts.append(f"{m} {100 * a['flip']['mean']:.1f}% ({a['n_flips']}/{a['n_calls']})")
    out.append(f"11. The A/A flip rate is under 3% for every model: {_v(ok if have else None)} ({'; '.join(parts)}).")

    # 12 thinking model vs qwen3:8b with distractors at 4k and two-hop far at 4k
    big, small = "qwen3:30b-a3b", "qwen3:8b"
    if big in grids:
        sub = df[(df.task == "key") & (df.variant == "distract") & (df.length == C.BASE_LENGTH) & (df.tag == "")]
        piv = sub[sub.model.isin([big, small])].pivot_table(index=["id", "depth"], columns="model", values="correct")
        piv = piv.dropna()
        if len(piv) and big in piv and small in piv:
            per = (piv[big] - piv[small]).groupby(level=0).mean()
            r = mean_ci(per.values)
            out.append(f"12. qwen3:30b-a3b (thinking) is more accurate than qwen3:8b with distractors at "
                       f"{C.BASE_LENGTH} on the same items (interval above zero): {_v(r['lo'] > 0)} ({_pp(r)} over "
                       f"{r['n']} items; confounded with size and thinking).")
        else:
            out.append("12. Thinking model: not testable (no shared items).")
    else:
        out.append("12. Thinking model: not testable (not run).")
    # 13 two-hop order: birthplace first is harder
    parts, n = [], 0
    for m in main:
        t = ths[m].get("4k")
        if not t:
            continue
        sub = df[(df.model == m) & (df.task == "twohop") & (df.tag == "") & (df.length == "4k")]
        r = paired_ci(sub[sub.variant == "far_back"], "correct", sub[sub.variant == "far_fwd"])
        n += _below(r)
        parts.append(f"{m} {_pp(r)}")
    out.append(f"13. Two-hop at 4k with the facts far apart is less accurate when the birthplace comes first and the "
               f"person's link to the entity last (interval below zero) for at least two main models: "
               f"{_v(n >= 2 if len(parts) >= 2 else None)} ({'; '.join(parts)}).")
    return [x + "\n" for x in out]

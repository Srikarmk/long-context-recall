"""reports/results.md and reports/figures/, generated from results/ and the item file only."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import analysis as A  # noqa: E402
from . import config as C  # noqa: E402
from .check import OUTCOMES  # noqa: E402
from .hypotheses import hypothesis_lines  # noqa: E402
from .items import load_items  # noqa: E402
from .prompts import Cell, render  # noqa: E402

MODEL_COLOR = {"qwen3:8b": "#2a78d6", "llama3.1:8b": "#eb6834", "gemma3:12b": "#1baf7a", "qwen3:30b-a3b": "#eda100"}
LEN_COLOR = {"1k": "#c9d9f2", "4k": "#8fb3e6", "8k": "#4f86d0", "16k": "#1f56a8", "32k": "#0b2c66"}
OUT_COLOR = {"correct": "#1baf7a", "mixed": "#9bd3bd", "wrong_entity": "#d64545", "bridge": "#eda100",
             "abstain": "#c9c6be", "other": "#52514e"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
DEPTH_LABEL = {0.0: "start", 0.25: "25%", 0.5: "middle", 0.75: "75%", 1.0: "end"}
PLACE_LABEL = {"near": "adjacent, middle", "far_fwd": "far: person at 10%, birthplace at 90%",
               "far_back": "far: birthplace at 10%, person at 90%"}
mlabel = A.mlabel
FIGDIR = {"path": C.FIGURES, "link": "figures"}


def pct(x, d: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def ci(r: dict, d: int = 0) -> str:
    if not r or not r.get("n"):
        return "n/a"
    return f"{pct(r['mean'], d)} [{100 * r['lo']:.{d}f}, {100 * r['hi']:.{d}f}]"


def pp(r: dict, d: int = 1) -> str:
    if not r or not r.get("n"):
        return "n/a"
    star = " *" if r["lo"] > 0 or r["hi"] < 0 else ""
    return f"{100 * r['mean']:+.{d}f} pp [{100 * r['lo']:+.{d}f}, {100 * r['hi']:+.{d}f}]{star}"


def md_table(header, rows) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows] + [""]


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(FIGDIR["path"], exist_ok=True)
    fig.savefig(os.path.join(FIGDIR["path"], name), dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _err(r):
    return [[100 * (r["mean"] - r["lo"])], [100 * (r["hi"] - r["mean"])]]


# ---------------------------------------------------------------- figures

def fig_depth(grids: dict, models: list[str], name: str = "depth_by_length.png", what: str = "one planted fact"):
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.3), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, m in zip(axes[0], models):
        style_axes(ax)
        g = grids[m]
        lens = [L for L in A.LEN_ORDER if (L, None) in g]
        for j, L in enumerate(lens):
            xs = [k + (j - (len(lens) - 1) / 2) * 0.06 for k in range(len(C.DEPTHS))]
            ys = [100 * g[(L, d)]["mean"] for d in C.DEPTHS]
            lo = [100 * (g[(L, d)]["mean"] - g[(L, d)]["lo"]) for d in C.DEPTHS]
            hi = [100 * (g[(L, d)]["hi"] - g[(L, d)]["mean"]) for d in C.DEPTHS]
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", color=LEN_COLOR[L], markersize=3.5, linewidth=1.3,
                        elinewidth=0.9, capsize=0, label=L)
        ax.set_xticks(range(len(C.DEPTHS)), [DEPTH_LABEL[d] for d in C.DEPTHS])
        ax.set_ylim(0, 103)
        ax.set_title(mlabel(m), fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("where the fact is in the document", fontsize=8, color=INK2)
    axes[0][0].set_ylabel("answered correctly (% of items)", fontsize=8, color=INK2)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, frameon=False, loc="lower center", ncol=len(lab), bbox_to_anchor=(0.5, -0.1),
               title="document length (tokens)", title_fontsize=7.5)
    fig.suptitle(f"Recall of {what} by depth and document length (95% CI over items)", fontsize=10,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, name)


def fig_length(grids: dict, dgrids: dict, models: list[str]):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for j, m in [(j, m) for j, m in enumerate(models)] + [(j, (m, "d")) for j, m in enumerate(models)]:
        dashed = isinstance(m, tuple)
        m = m[0] if dashed else m
        g = (dgrids if dashed else grids)[m]
        lens = [L for L in A.LEN_ORDER if (L, None) in g and g[(L, None)]["n"]]
        xs = [A.LEN_ORDER.index(L) + (j - (len(models) - 1) / 2) * 0.05 for L in lens]
        ys = [100 * g[(L, None)]["mean"] for L in lens]
        lo = [100 * (g[(L, None)]["mean"] - g[(L, None)]["lo"]) for L in lens]
        hi = [100 * (g[(L, None)]["hi"] - g[(L, None)]["mean"]) for L in lens]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="--s" if dashed else "-o", color=MODEL_COLOR[m], markersize=4,
                    linewidth=1.4, elinewidth=1.0, capsize=0, markerfacecolor=SURFACE if dashed else MODEL_COLOR[m],
                    label=f"{mlabel(m)}, with distractors" if dashed else mlabel(m))
    ax.set_xticks(range(len(A.LEN_ORDER)), A.LEN_ORDER)
    ax.set_ylim(0, 103)
    ax.set_xlabel("document length (tokens, nominal)", fontsize=8, color=INK2)
    ax.set_ylabel("answered correctly (% of items, all depths)", fontsize=8, color=INK2)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_title("Recall by document length: solid = fact alone, dashed = fact with three near misses", fontsize=9,
                 color=INK, loc="left")
    save(fig, "length.png")


def fig_outcomes(dt: dict, th: dict, models: list[str]):
    """Stacked outcome shares: key fact at 4k with distractors, fact absent, two-hop far."""
    cols = []
    for m in models:
        d = dt.get(m) or {}
        if d:
            cols.append((m, "fact + distractors", d["shares"]))
            for v, lab in (("plain", "fact absent"), ("distract", "absent + distractors")):
                if v in d.get("absent", {}):
                    cols.append((m, lab, d["absent"][v]["shares"]))
        t = th.get(m) or {}
        for v in ("near", "far_fwd", "far_back"):
            if t and v in t["shares"]:
                cols.append((m, f"two-hop {v.replace('_', ' ')}", t["shares"][v]))
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(7.6, 0.3 * len(cols) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax, "x")
    for i, (m, lab, sh) in enumerate(cols):
        left = 0
        for o in OUTCOMES:
            w = 100 * sh[o]["mean"] if sh[o]["n"] else 0
            ax.barh(i, w, left=left, color=OUT_COLOR[o], height=0.66, label=o if i == 0 else None)
            left += w
    ax.set_yticks(range(len(cols)), [f"{mlabel(m)}: {lab}" for m, lab, _ in cols], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of calls ({C.BASE_LENGTH} documents)", fontsize=8, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.32 - 0.6 / len(cols)))
    ax.set_title("What the reply named. For `fact absent` the right reply is to abstain", fontsize=9, color=INK,
                 loc="left")
    save(fig, "outcomes.png")


# ---------------------------------------------------------------- report

def write_report(split: str = "main") -> None:
    df = A.load_rows(split)
    if df.empty:
        raise SystemExit("no results yet")
    models = A.models_in(df)
    grids = {m: A.grid(df, m) for m in models}
    dgrids = {m: A.grid(df, m, "distract") for m in models}
    le = {m: A.length_effect(df, m) for m in models}
    dle = {m: A.length_effect(df, m, "distract") for m in models}
    dts = {m: {L: A.distractor_tables(df, m, L) for L in A.LEN_ORDER} for m in models}
    dt = {m: dts[m][C.BASE_LENGTH] for m in models}
    ths = {m: {L: A.twohop_tables(df, m, L) for L in ("4k", "16k")} for m in models}
    th = {m: ths[m]["4k"] for m in models}
    aa = {m: A.aa_tables(df, m) for m in models}
    cb = {m: A.closed_tables(df, m) for m in models}
    facts = A.run_facts(df)
    meta_path = os.path.join(C.RESULTS, "models.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if split == "pilot":
        FIGDIR.update(path=os.path.join(C.REPORTS, "pilot_figures"), link="pilot_figures")
    else:
        FIGDIR.update(path=C.FIGURES, link="figures")
    fl = FIGDIR["link"]
    fig_depth(grids, models)
    fig_depth(dgrids, models, "depth_by_length_distract.png", "the fact among three near misses")
    fig_length(grids, dgrids, models)
    fig_outcomes(dt, th, models)

    L = ["# Results", "",
         f"Generated by `uv run lcrecall report{' --split pilot' if split == 'pilot' else ''}` from "
         "`results/rows/*.jsonl` and `data/items.jsonl`. Every number below is computed by code from cached model "
         "outputs, re-checked with the current answer checker. Intervals are 95% percentile bootstraps "
         f"({C.BOOT:,} resamples) over items: every cell of an item holds the same fictional fact, so an item's "
         "cells move together and differences between cells are paired by item. `pp` = percentage points; `*` "
         "marks a difference whose interval excludes zero.", ""]
    if split == "pilot":
        L += ["**Pilot split.** These are the pilot items, used only to check prompts, the checker, the token "
              "check and timing. They are excluded from every reported result.", ""]

    # headline
    L += ["## Headline", ""]
    for m in models:
        parts = []
        for name, g in (("fact alone", grids[m]), ("with distractors", dgrids[m])):
            got = [f"{Ln} {ci(g[(Ln, None)])}" for Ln in A.LEN_ORDER if (Ln, None) in g and g[(Ln, None)]["n"]]
            if got:
                parts.append(f"{name}: " + "; ".join(got))
        line = f"- **{mlabel(m)}**, recall over all depths, " + ". ".join(parts) + "."
        t, a = th.get(m), aa.get(m)
        if t:
            line += (f" Two-hop at 4k: adjacent {ci(t['acc']['near'])}, far "
                     f"{ci(t['acc']['far_fwd'])} / {ci(t['acc']['far_back'])}.")
        if a:
            line += f" A/A flip rate {ci(a['flip'], 1)}."
        L.append(line)
    L.append("")

    # 1 setup
    L += ["## 1. Setup and token check", "",
          f"Every call uses num_ctx {C.NUM_CTX:,}. `ratio` = the server's prompt_eval_count over the estimate "
          f"(characters / chars-per-token + template tokens); a call outside [{C.TOKEN_LOW}, {C.TOKEN_HIGH}] "
          "or over num_ctx would have stopped the run (Ollama shortens an over-long prompt silently), so every "
          "row below passed. `wall` = median seconds per call, including prefix reuse within an item.", ""]
    rows = []
    for f in facts:
        info = meta.get(f["model"], {})
        rows.append([mlabel(f["model"]), info.get("digest", "")[:12], f["length"], f["calls"], f["items"],
                     f"{f['prompt_tokens']:,.0f}", f"{f['max_tokens']:,}", f"{f['ratio_lo']:.2f} to {f['ratio_hi']:.2f}",
                     f"{f['wall']:.1f}", f"{f['prefill_tps']:,.0f}", f"{f['output_tokens']:.0f}",
                     f"{f['thinking_chars']:,.0f}", pct(f["truncated"])])
    L += md_table(["model", "digest", "length", "calls", "items", "mean prompt tokens", "max", "ratio", "median wall s/call",
                   "median prefill tok/s", "output tokens", "thinking chars", "hit answer cap"], rows)

    # 2 closed book
    L += ["## 2. Closed-book control", "",
          "The question with no document. Every answer is invented, so a correct reply here would mean the "
          "answer leaked into the question or the model guessed it.", ""]
    rows = []
    for m in models:
        for v, r in (cb.get(m) or {}).items():
            sh = r["shares"]
            rows.append([mlabel(m), "key question" if v == "key" else "second-hop question", r["n"],
                         ci(sh["correct"]), ci(sh["abstain"]), ci(sh["wrong_entity"]), ci(sh["other"])])
    L += md_table(["model", "question", "calls", "correct", "abstain", "near-miss name", "other name"], rows) \
        if rows else ["Not run yet.", ""]

    # 3 grid
    L += ["## 3. Recall by length and depth", "",
          "Share of items answered correctly with the key fact at each depth, alone and with the three near-miss "
          "distractors. `all` averages an item's five depths.", "", f"![length]({fl}/length.png)", "",
          f"![depth]({fl}/depth_by_length.png)", "", f"![depth, distractors]({fl}/depth_by_length_distract.png)", ""]
    for m in models:
        for name, g in (("fact alone", grids[m]), ("with distractors", dgrids[m])):
            rows = [[Ln, g[(Ln, 0.0)]["n"]] + [ci(g[(Ln, d)]) for d in C.DEPTHS] + [ci(g[(Ln, None)])]
                    for Ln in A.LEN_ORDER if (Ln, None) in g]
            if rows:
                L += [f"**{mlabel(m)}, {name}**", ""]
                L += md_table(["length", "items"] + [DEPTH_LABEL[d] for d in C.DEPTHS] + ["all"], rows)

    # 4 length
    L += ["## 4. Effect of length", "",
          "Accuracy over all depths at each length minus at 1k, paired by item (the longer document contains "
          "the shorter one as its first part).", ""]
    rows = [[mlabel(m), name] + [pp(e.get(Ln, {})) for Ln in A.LEN_ORDER[1:]]
            for m in models for name, e in (("fact alone", le[m]), ("with distractors", dle[m])) if e]
    L += md_table(["model", "condition"] + [f"{Ln} minus 1k" for Ln in A.LEN_ORDER[1:]], rows)

    # 5 position
    L += ["## 5. Effect of position", "",
          "Middle minus ends = accuracy with the fact in the middle minus the mean of start and end; interior "
          "minus ends uses the mean of 25%, 50% and 75%; start minus end compares the two edges. Paired by "
          "item. `1k-8k` pools 1k, 4k and 8k (an item's lengths averaged); `16k-32k` pools whatever ran.", ""]
    rows = []
    for m in models:
        for variant, vname in (("plain", "alone"), ("distract", "distractors")):
            for lab, lens in [(Ln, [Ln]) for Ln in A.LEN_ORDER] + [("1k-8k", A.SHORT), ("16k-32k", ["16k", "32k"])]:
                p = A.position_effect(df, m, lens, variant)
                if p:
                    rows.append([mlabel(m), vname, lab, p["n"], pp(p["middle_minus_ends"]), pp(p["interior_minus_ends"]),
                                 pp(p["start_minus_end"]), f"{DEPTH_LABEL[p['worst_depth']]} ({pct(p['worst'], 0)})"])
    L += md_table(["model", "fact", "lengths", "items", "middle minus ends", "interior minus ends", "start minus end",
                   "worst depth"], rows)

    # 6 distractors
    L += ["## 6. Near-miss distractors and missing facts", "",
          "Three near-miss facts (the same name with another type of entity, the same type with the name's "
          "ending changed, the same type with its beginning changed), each naming a different person, at "
          "seeded depths away from the key positions. Effect = accuracy with distractors minus without, same "
          f"items and depths. Fact absent (at {C.BASE_LENGTH}): the key fact is left out, so the right reply is "
          "`not stated`.", "", f"![outcomes]({fl}/outcomes.png)", ""]
    rows = []
    for m in models:
        for Ln, d in dts[m].items():
            if not d:
                continue
            rows.append([mlabel(m), Ln, "all", d["n_items"], ci(d["acc_plain"]), ci(d["acc_distract"]), pp(d["effect"])])
            for dep, r in d["by_depth"].items():
                rows.append([mlabel(m), Ln, DEPTH_LABEL[dep], r["distract"]["n"], ci(r["plain"]), ci(r["distract"]),
                             pp(r["effect"])])
    L += md_table(["model", "length", "depth", "items", "without distractors", "with distractors", "effect"], rows) \
        if rows else ["Not run yet.", ""]
    rows = []
    for m in models:
        for Ln, d in dts[m].items():
            if not d:
                continue
            sh = d["shares"]
            rows.append([mlabel(m), Ln, "fact + distractors", d["n_calls"]] + [ci(sh[o]) for o in OUTCOMES]
                        + [", ".join(f"{k} {v}" for k, v in d["picked"].items())])
            for v, r in d.get("absent", {}).items():
                rows.append([mlabel(m), Ln, "fact absent" + (" + distractors" if v == "distract" else ""), r["n"]]
                            + [ci(r["shares"][o]) for o in OUTCOMES] + [""])
    if rows:
        L += ["Outcome shares (`wrong_entity` = named a distractor's person):", ""]
        L += md_table(["model", "length", "condition", "calls"] + OUTCOMES + ["near misses picked (calls)"], rows)
    rows = []
    for m in models:
        pdp = A.picked_depth(df, m)
        if pdp:
            rows.append([mlabel(m), pdp["n"], f"{pdp['before']} ({pct(pdp['before'] / pdp['n'], 0)})",
                         pct(pdp["chance_before"], 0), f"{pdp['nearest']} ({pct(pdp['nearest'] / pdp['n'], 0)})"])
    if rows:
        L += ["Where the named distractor was, over all lengths. Chance = the share expected if the reply named one of "
              "the three distractors at random (it depends on where the key fact is); by chance a third are nearest.", ""]
        L += md_table(["model", "wrong-entity replies", "distractor before the key fact", "chance", "nearest to the key fact"],
                      rows)

    # 7 two-hop
    L += ["## 7. Two-hop: combining two facts", "",
          "\"In which town was the founder of X born?\" needs the key fact (X to a person) and the person's "
          "birthplace; two other people's birthplaces are decoys at 30% and 70%. Second hop alone = \"In which "
          "town was <person> born?\" in the far-forward document. Single fact = the key question at the same "
          "length without distractors (all depths).", ""]
    rows, rows2 = [], []
    for m in models:
        for Ln, t in ths[m].items():
            if not t:
                continue
            for v in TWOHOP:
                sh = t["shares"][v]
                rows.append([mlabel(m), Ln, PLACE_LABEL[v], t["acc"][v]["n"], ci(t["acc"][v]),
                             pp(t["vs_near"][v]) if v in t["vs_near"] else "", ci(sh["wrong_entity"]),
                             ci(sh["bridge"]), ci(sh["abstain"]), ci(sh["other"])])
            rows2.append([mlabel(m), Ln, ci(t["single"]), ci(t["hop2"]), ci(t["acc"]["far_fwd"]), pp(t["fwd_minus_hop2"])])
    L += md_table(["model", "length", "placement", "items", "correct", "minus adjacent", "decoy town",
                   "named the person only", "abstain", "other"], rows) if rows else ["Not run yet.", ""]
    if rows2:
        L += md_table(["model", "length", "single fact", "second hop alone", "two-hop far forward",
                       "two-hop minus second hop"], rows2)

    # 8 A/A
    L += ["## 8. Noise floor: the identical prompt sent twice", "",
          "Each main item's five with-distractor cells at one length (1k, 4k or 8k in rotation) are sent again as "
          "separate calls with their own cache entries, in a later stage. These cells are repeated because their "
          "answers vary; cells at ceiling would show no flips by construction. Flip = correctness differs from "
          "the first call. An effect counts only when its interval excludes zero and it is larger than the "
          "model's flip rate.", ""]
    rows = []
    for m in models:
        a = aa.get(m)
        if a:
            rows.append([mlabel(m), a["n_items"], a["n_calls"], a["n_flips"], ci(a["flip"], 1), ci(a["text_changed"], 1),
                         ci(a["raw_changed"], 1)] + [f"{a['by_length'][Ln]['flips']}/{a['by_length'][Ln]['n']}"
                                                     if Ln in a["by_length"] else "" for Ln in A.SHORT])
    L += md_table(["model", "items", "repeated calls", "flips", "flip rate", "answer changed", "raw text changed"]
                  + [f"flips at {Ln}" for Ln in A.SHORT], rows) if rows else ["Not run yet.", ""]

    # 9 predictions
    if split == "main":
        L += ["## 9. Pre-registered predictions", "", "Scored against `HYPOTHESES.md` (written before the full run).", ""]
        L += hypothesis_lines(df, grids, dgrids, le, dle, dts, ths, aa, cb)

    # 10 sample prompt
    L += ["## 10. Sample rendered prompts", "", "The document is cut to the parts around the planted facts.", ""]
    items = load_items(split)
    if items:
        it = items[0]
        for cell in (Cell("key", "1k", 0.5, "distract"), Cell("twohop", "1k", None, "far_fwd")):
            text = render(it, cell).messages[-1]["content"]
            marks = [it["needle"], it["hop2"]["text"]] + [d["text"] for d in it["distractors"] + it["decoys"]]
            spans = sorted(text.find(x) for x in marks if x in text)
            parts = [text[:330]] + [text[max(0, s - 120):s + 330] for s in spans] + [text[-330:]]
            L += [f"**{it['id']}, {cell.key}**", "", "```", "\n[...]\n".join(parts), "```", ""]

    # 11 hand-check
    L += ["## 11. Hand-check of the answer checker", ""]
    pre = "" if split == "main" else "pilot_"
    ag = os.path.join(C.HANDCHECK, f"{pre}agreement.json")
    if os.path.exists(ag):
        L += [json.load(open(ag))["summary"], ""]
    else:
        L += [f"`handcheck/{pre}sample.csv` holds a seeded sample of outputs with an empty `label` column. It has "
              "not been labelled yet; once Srikar labels it, `uv run lcrecall handcheck score` writes the agreement "
              "here.", ""]

    os.makedirs(C.REPORTS, exist_ok=True)
    name = "results.md" if split == "main" else "pilot.md"
    with open(os.path.join(C.REPORTS, name), "w") as f:
        f.write("\n".join(L).rstrip() + "\n")
    print(f"wrote reports/{name} and figures")


TWOHOP = ["near", "far_fwd", "far_back"]

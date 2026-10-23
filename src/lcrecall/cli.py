"""lcrecall command line."""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import config as C


def _client(args):
    from .llm import Cache, Client
    os.makedirs(os.path.dirname(C.CACHE_DB), exist_ok=True)
    return Client(Cache(C.CACHE_DB), cache_only=getattr(args, "cache_only", False))


def cmd_fetch(args):
    from .data import corpus_facts, fetch
    for name, meta in fetch().items():
        print(f"{name}: {meta['repo']}@{meta['revision'][:10]} {meta['path']} {meta['bytes']} bytes")
    print("filler:", corpus_facts())


def cmd_items(args):
    from collections import Counter

    from .items import write_items
    items = write_items()
    print(f"wrote {len(items)} items to data/items.jsonl:", dict(Counter(i["split"] for i in items)),
          dict(Counter(i["relation"] for i in items if i["split"] == "main")))


def _record_model(model: str):
    from .llm import model_info, think_flag
    path = os.path.join(C.RESULTS, "models.json")
    meta = {}
    if os.path.exists(path):
        with open(path) as f:
            meta = json.load(f)
    meta[model] = {**model_info(model), "think": think_flag(model), "options": C.options(model),
                   "native_ctx": C.NATIVE_CTX[model], "token_cpt": C.TOKEN_CPT[model]}
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)


def _plan(args, model: str) -> list[tuple[str, int]]:
    """(stage, first N items) for one model under --grid, or --stage with --n-items."""
    from .runner import STAGE_ORDER
    if args.grid:
        grid = C.THINK_GRID if model in C.THINKING_ONLY else C.GRID
        return [(st, grid[st]) for st in STAGE_ORDER if st in grid]
    stages = STAGE_ORDER if args.stage == "all" else args.stage.split(",")
    for st in stages:
        if st not in STAGE_ORDER:
            raise SystemExit(f"unknown stage {st}")
    return [(st, args.n_items) for st in STAGE_ORDER if st in stages]


def cmd_run(args):
    """Stage by stage; within a stage one model at a time (long prompts are compute-bound, so models side
    by side would not finish sooner and would muddle the timings). --late-models run their own plan after
    every stage of --models is done."""
    from .runner import STAGE_ORDER, run_stage
    client = _client(args)
    models = [m for m in args.models.split(",") if m]
    late = [m for m in args.late_models.split(",") if m]
    if not args.cache_only:
        for m in models + late:
            _record_model(m)
    for group in (models, late):
        plans = {m: dict(_plan(args, m)) for m in group}
        for st in STAGE_ORDER:
            for m in group:
                if st not in plans[m]:
                    continue
                n = plans[m][st]
                rows = run_stage(st, client, m, split=args.split, n_items=n, skip_missing=args.skip_missing)
                size = f"first {n} items" if n else "all items"
                print(f"{m} {st} ({size}): {len(rows)} rows", flush=True)
    print("cache", client.stats, flush=True)


def cmd_plan(args):
    """Calls per stage and model for the planned grid, and hours at the pilot's seconds per call."""
    from .analysis import pilot_rates
    from .runner import jobs_for
    rates = pilot_rates()
    total = 0.0
    for m in args.models.split(","):
        grid = C.THINK_GRID if m in C.THINKING_ONLY else C.GRID
        calls, secs, parts = 0, 0.0, []
        for st, n in grid.items():
            jobs = jobs_for(st, args.split, n)
            s = sum(rates.get((m, st), rates.get((m, c.length or "closed"), float("nan"))) for _, c in jobs)
            calls += len(jobs)
            secs += s
            parts.append(f"{st}={len(jobs)} ({s / 3600:.2f} h)")
        total += secs
        print(f"{m}: {calls} calls, {secs / 3600:.2f} h: " + ", ".join(parts))
    print(f"total {total / 3600:.2f} h")


def cmd_show(args):
    """Print the rendered prompt of one cell (document shortened), and the stored output if any."""
    from .items import load_items
    from .prompts import Cell, render
    from .runner import STAGES, read_rows
    it = next(i for i in load_items(None) if i["id"] == args.id)
    rows = {r["cell"]: r for r in read_rows(args.model, it["split"]) if r["id"] == it["id"]}
    cells = [Cell.from_key(args.cell)] if args.cell else [c for st in STAGES.values() for c in st(it)]
    for c in cells:
        rd = render(it, c)
        text = rd.messages[-1]["content"]
        if not args.full and len(text) > 3000:
            marks = [it["needle"], it["hop2"]["text"]] + [d["text"] for d in it["distractors"] + it["decoys"]]
            spans = sorted(text.find(m) for m in marks if m in text)
            parts = [text[:400]] + [text[max(0, s - 200):s + 400] for s in spans] + [text[-450:]]
            text = "\n[...]\n".join(parts)
        print(f"===== {c.key} reached={rd.reached} doc_chars={rd.doc_chars} gold={rd.gold} wrong={rd.wrong}")
        print(text)
        r = rows.get(c.key)
        if r:
            print(f"--- {args.model}: outcome={r['outcome']} prompt_tokens={r['prompt_tokens']} "
                  f"seconds={r['seconds']:.1f}\n{r['output']}")
        print()


def cmd_report(args):
    from .report import write_report
    write_report(split=args.split)


def cmd_handcheck(args):
    from . import handcheck
    if args.action == "sample":
        handcheck.sample(args.n, split=args.split)
    else:
        handcheck.score(split=args.split)


def main(argv=None):
    p = argparse.ArgumentParser(prog="lcrecall")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch", help="download the pinned data file")
    sub.add_parser("items", help="build data/items.jsonl")
    r = sub.add_parser("run", help="run stages")
    r.add_argument("--grid", action="store_true", help="the planned grid in config.GRID / config.THINK_GRID")
    r.add_argument("--stage", default="all", help="comma-separated stages, or all (ignored with --grid)")
    r.add_argument("--models", default=",".join(C.MAIN_MODELS))
    r.add_argument("--late-models", default="", help="models that run after every stage of --models")
    r.add_argument("--split", default="main", choices=["main", "pilot"])
    r.add_argument("--n-items", type=int, default=0, help="only the first N items")
    r.add_argument("--cache-only", action="store_true")
    r.add_argument("--skip-missing", action="store_true", help="with --cache-only, skip calls not in the cache")
    pl = sub.add_parser("plan", help="count the calls of the planned grid and estimate hours from the pilot")
    pl.add_argument("--split", default="main", choices=["main", "pilot"])
    pl.add_argument("--models", default=",".join(C.MODELS))
    s = sub.add_parser("show", help="print an item's rendered prompts and stored outputs")
    s.add_argument("id")
    s.add_argument("--cell", default="", help="a cell key such as 'key|4k|0.50|plain|'")
    s.add_argument("--model", default=C.MODELS[0])
    s.add_argument("--full", action="store_true", help="print the whole document")
    rp = sub.add_parser("report", help="write reports/results.md (or pilot.md) and figures")
    rp.add_argument("--split", default="main", choices=["main", "pilot"])
    hc = sub.add_parser("handcheck", help="sample outputs for labelling, or score the labels")
    hc.add_argument("action", choices=["sample", "score"])
    hc.add_argument("--n", type=int, default=60)
    hc.add_argument("--split", default="main", choices=["main", "pilot"])
    args = p.parse_args(argv)
    fn = {"fetch": cmd_fetch, "items": cmd_items, "run": cmd_run, "plan": cmd_plan, "show": cmd_show,
          "report": cmd_report, "handcheck": cmd_handcheck}[args.cmd]
    return fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())

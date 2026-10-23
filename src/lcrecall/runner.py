"""Send every call of a stage to a model, check it, and append one row per call.

The run is split into stages. Each stage runs for every model before the next stage starts, in this
order, so that a cut leaves the most useful part done:

  closed        the key question and the second-hop question with no document  (closed-book control)
  short         key fact at 5 depths in 1k, 4k and 8k documents                  (length x position)
  distract      key fact at 5 depths in 4k documents with 3 near-miss
                distractors, and the fact left out (with and without them)      (distractors, abstention)
  twohop        two-hop question at 4k with the two facts adjacent or far apart
                (both orders), and the second hop alone                         (composition)
  distract_len  key fact with distractors at 5 depths in 1k and 8k documents    (distractors x length)
  aa            each item's 5 distractor cells at one of 1k, 4k, 8k sent again  (noise floor)
  long16        key fact at 5 depths in 16k documents, without and with distractors
  twohop16      the two-hop calls at 16k
  long32        key fact at 5 depths in 32k documents, without and with distractors

Within a stage the calls of an item are sent one after another, sorted by prompt text: the server keeps
the previous prompt's cache and reuses the longest shared prefix, and sorted prompts share the longest
prefixes. This changes only the speed. Rows are appended as they finish, so an interrupted stage keeps
what it did; every call is also in the cache, so re-running a stage only sends the calls that are
missing.
"""

from __future__ import annotations

import json
import os
import time

from . import config as C
from .check import classify
from .items import load_items
from .llm import CacheMiss, Client
from .prompts import TWOHOP_PLACEMENTS, Cell, render

B = C.BASE_LENGTH


def key_cells(lengths, variant="plain", tag=""):
    return [Cell("key", L, d, variant, tag) for L in lengths for d in C.DEPTHS]


def twohop_cells(L):
    return [Cell("twohop", L, None, v) for v in TWOHOP_PLACEMENTS] + [Cell("hop2", L, None, "far_fwd")]


STAGES = {
    "closed": lambda it: [Cell("closed", variant="key"), Cell("closed", variant="hop2")],
    "short": lambda it: key_cells(["1k", "4k", "8k"]),
    "distract": lambda it: key_cells([B], "distract") + [Cell("absent", B, None, "plain"),
                                                         Cell("absent", B, None, "distract")],
    "twohop": lambda it: twohop_cells(B),
    "distract_len": lambda it: key_cells(["1k", "8k"], "distract"),
    "aa": lambda it: key_cells([it["aa_length"]], "distract", tag="aa"),
    "long16": lambda it: key_cells(["16k"]) + key_cells(["16k"], "distract"),
    "twohop16": lambda it: twohop_cells("16k"),
    "long32": lambda it: key_cells(["32k"]) + key_cells(["32k"], "distract"),
}
STAGE_ORDER = list(STAGES)
assert [s for s in STAGE_ORDER if s in C.GRID] == list(C.GRID) and set(C.THINK_GRID) <= set(STAGES)


class TokenCheckError(RuntimeError):
    """The prompt would not fit, or the server evaluated a different number of tokens than were sent."""


def slug(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def rows_path(model: str, split: str = "main") -> str:
    name = slug(model) + ("" if split == "main" else f"_{split}")
    return os.path.join(C.RESULTS, "rows", f"{name}.jsonl")


def read_rows(model: str, split: str = "main") -> list[dict]:
    """Rows of one model, the last row kept for each (item, cell)."""
    path = rows_path(model, split)
    if not os.path.exists(path):
        return []
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(r["id"], r["cell"])] = r
    return list(out.values())


def est_tokens(model: str, chars: int) -> float:
    return chars / C.TOKEN_CPT[model] + C.TEMPLATE_TOKENS[model]


def check_before(model: str, chars: int, options: dict) -> None:
    est = est_tokens(model, chars)
    if est + options["num_predict"] > options["num_ctx"] or est > C.NATIVE_CTX[model]:
        raise TokenCheckError(f"{model}: about {est:.0f} prompt tokens + {options['num_predict']} do not fit "
                              f"num_ctx {options['num_ctx']} (native window {C.NATIVE_CTX[model]})")


def check_after(model: str, chars: int, prompt_tokens: int, options: dict) -> None:
    est = est_tokens(model, chars)
    ratio = prompt_tokens / est if est else 0
    if not (C.TOKEN_LOW <= ratio <= C.TOKEN_HIGH):
        raise TokenCheckError(f"{model}: server evaluated {prompt_tokens} prompt tokens, expected about {est:.0f} "
                              f"(ratio {ratio:.2f}); the prompt was probably truncated")
    if prompt_tokens + options["num_predict"] > options["num_ctx"] or prompt_tokens > C.NATIVE_CTX[model]:
        raise TokenCheckError(f"{model}: {prompt_tokens} prompt tokens + {options['num_predict']} exceed num_ctx "
                              f"{options['num_ctx']} or the native window {C.NATIVE_CTX[model]}")


class Progress:
    def __init__(self, total: int, label: str, every: int = 25):
        self.total, self.label, self.every = total, label, every
        self.done, self.t0 = 0, time.time()

    def tick(self, client: Client):
        self.done += 1
        if self.done % self.every == 0 or self.done == self.total:
            el = time.time() - self.t0
            print(f"[{self.label}] {self.done}/{self.total} new_calls={client.stats['misses']} "
                  f"hits={client.stats['hits']} elapsed={el:.0f}s", flush=True)


def first_n(items: list[dict], n: int) -> list[dict]:
    return items if not n else items[:n]


def jobs_for(stage: str, split: str = "main", n_items: int = 0) -> list[tuple[dict, Cell]]:
    jobs = []
    for it in first_n(load_items(split), n_items):
        cells = STAGES[stage](it)
        cells.sort(key=lambda c: render(it, c).messages[-1]["content"])
        jobs += [(it, c) for c in cells]
    return jobs


def score_row(it: dict, cell: Cell, text: str) -> dict:
    """The checked fields of one output; also used to re-check stored rows with the current checker."""
    rd = render(it, cell)
    return classify(text, rd.gold, rd.wrong, rd.bridge)


def run_one(client: Client, model: str, it: dict, cell: Cell, stage: str) -> dict:
    rd = render(it, cell)
    opts = C.options(model)
    chars = sum(len(m["content"]) for m in rd.messages)
    check_before(model, chars, opts)
    out = client.chat(model, rd.messages, opts, tag=cell.tag)
    check_after(model, chars, out["prompt_tokens"], opts)
    row = {"model": model, "id": it["id"], "split": it["split"], "stage": stage, "cell": cell.key,
           "task": cell.task, "length": cell.length, "depth": cell.depth, "variant": cell.variant, "tag": cell.tag,
           "output": out["text"], "truncated": out.get("done_reason") == "length",
           "prompt_chars": chars, "prompt_tokens": out["prompt_tokens"], "est_tokens": round(est_tokens(model, chars)),
           "output_tokens": out["output_tokens"], "seconds": out["seconds"],
           "prefill_seconds": out.get("prefill_seconds"), "wall_seconds": out.get("wall_seconds"),
           "cached": out["cached"], "thinking_chars": len(out.get("thinking") or ""), "reached": rd.reached}
    row.update(classify(out["text"], rd.gold, rd.wrong, rd.bridge))
    return row


def run_stage(stage: str, client: Client, model: str, split: str = "main", n_items: int = 0,
              skip_missing: bool = False) -> list[dict]:
    jobs = jobs_for(stage, split, n_items)
    if not jobs:
        print(f"[{model} {stage}] nothing to run", flush=True)
        return []
    prog = Progress(len(jobs), f"{model} {stage}")
    path = rows_path(model, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for it, cell in jobs:
        try:
            row = run_one(client, model, it, cell, stage)
        except CacheMiss:
            if not skip_missing:
                raise
            row = None
        if row is not None:
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
        prog.tick(client)
    return rows

"""Prompt templates and the rendering of one cell (item x task x length x placement) into messages.

Tasks:
  key      the single-fact question; the key fact is placed at `depth` (variant `plain`), or at `depth`
           with the three near-miss distractors at their own depths (variant `distract`)
  absent   the single-fact question with the key fact left out: no fact at all (`plain`) or only the
           distractors (`distract`); the right reply is `not stated`
  twohop   "In which town was the founder of X born?": the key fact and the birthplace fact are adjacent
           in the middle (`near`), key at 10% and birthplace at 90% (`far_fwd`), or the reverse
           (`far_back`); two decoy birthplaces of other people sit at 30% and 70%
  hop2     "In which town was <person> born?" in the `far_fwd` two-hop document: the second hop alone
  closed   the key question (variant `key`) or the second-hop question (`hop2`) with no document
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .context import document, place

DOC_PROMPT = """Below is a long document, followed by a question about it. Answer the question using only the document.

<document>
{doc}
</document>

Question: {question}
Reply with only the answer, in a few words, and no explanation. If the document does not give the answer, reply exactly: not stated"""

CLOSED_PROMPT = """Answer the question below.

Question: {question}
Reply with only the answer, in a few words, and no explanation. If you do not know the answer, reply exactly: unknown"""

TWOHOP_PLACEMENTS = {"near": (0.5, 0.5), "far_fwd": (0.1, 0.9), "far_back": (0.9, 0.1)}   # (key, birthplace)


@dataclass(frozen=True)
class Cell:
    task: str
    length: str = ""
    depth: float | None = None
    variant: str = "plain"
    tag: str = ""            # not sent to the model: "aa" gives the A/A repeat its own cache entry

    @property
    def key(self) -> str:
        d = "" if self.depth is None else f"{self.depth:.2f}"
        return f"{self.task}|{self.length}|{d}|{self.variant}|{self.tag}"

    @staticmethod
    def from_key(k: str) -> "Cell":
        task, length, d, variant, tag = k.split("|")
        return Cell(task, length, float(d) if d else None, variant, tag)


@dataclass
class Rendered:
    messages: list[dict]
    gold: list[str] | None            # words that mark the right answer (None: the right reply is to abstain)
    wrong: dict[str, list[str]]       # near-miss answers: label -> marking words
    bridge: list[str] = field(default_factory=list)   # two-hop: the intermediate person
    reached: dict = field(default_factory=dict)       # depth actually reached by each placed fact
    doc_chars: int = 0


def _distractor_inserts(it: dict) -> list[tuple[float, str]]:
    return [(d["depth"], d["text"]) for d in it["distractors"]]


def answers(it: dict, task: str) -> tuple[list[str] | None, dict, list[str]]:
    person = [it["answer"]["last"]]
    dis = {d["kind"]: [d["person"]["last"]] for d in it["distractors"]}
    town = [it["hop2"]["town"]]
    decoys = {f"decoy{j + 1}": [d["town"]] for j, d in enumerate(it["decoys"])}
    if task == "key":
        return person, dis, []
    if task == "absent":
        return None, dis, []
    if task == "twohop":
        return town, decoys, person
    if task == "hop2":
        return town, decoys, []
    raise ValueError(task)


def render(it: dict, cell: Cell) -> Rendered:
    if cell.task == "closed":
        q = it["question"] if cell.variant == "key" else it["hop2_question"]
        gold, wrong, _ = answers(it, "key" if cell.variant == "key" else "hop2")
        return Rendered([{"role": "user", "content": CLOSED_PROMPT.format(question=q)}], gold, wrong)
    doc = document(it["filler_start"], cell.length)
    inserts, labels = [], []
    if cell.task in ("key", "absent"):
        if cell.variant == "distract":
            inserts += _distractor_inserts(it)
            labels += [d["kind"] for d in it["distractors"]]
        elif cell.variant != "plain":
            raise ValueError(cell)
        if cell.task == "key":
            inserts.append((cell.depth, it["needle"]))
            labels.append("key")
        question = it["question"]
    elif cell.task in ("twohop", "hop2"):
        k, h = TWOHOP_PLACEMENTS[cell.variant]
        inserts += [(d["depth"], d["text"]) for d in it["decoys"]]
        labels += ["decoy1", "decoy2"]
        if k <= h:
            inserts += [(k, it["needle"]), (h, it["hop2"]["text"])]
            labels += ["key", "hop2"]
        else:
            inserts += [(h, it["hop2"]["text"]), (k, it["needle"])]
            labels += ["hop2", "key"]
        question = it["twohop_question"] if cell.task == "twohop" else it["hop2_question"]
    else:
        raise ValueError(cell)
    text, reached = place(doc, inserts)
    gold, wrong, bridge = answers(it, cell.task)
    content = DOC_PROMPT.format(doc=text, question=question)
    return Rendered([{"role": "user", "content": content}], gold, wrong, bridge, dict(zip(labels, reached)), len(text))

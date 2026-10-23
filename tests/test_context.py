import pytest

from lcrecall import config as C
from lcrecall.context import boundaries, document, place
from lcrecall.items import build_items, check_items
from lcrecall.prompts import TWOHOP_PLACEMENTS, Cell, render
from lcrecall.runner import STAGES


@pytest.fixture(scope="module")
def items():
    return build_items()


def test_items_are_deterministic_and_valid(items):
    check_items(items)
    again = build_items()
    assert [i["needle"] for i in again] == [i["needle"] for i in items]
    assert sum(i["split"] == "pilot" for i in items) == C.N_PILOT
    assert len({i["name"] for i in items}) == len(items)


def test_documents_nest_and_fit_their_budget(items):
    it = items[0]
    prev = ""
    for L in C.LENGTHS:
        d = document(it["filler_start"], L)
        assert len(d) <= C.doc_chars(L) and len(d) > 0.9 * C.doc_chars(L)
        assert d.startswith(prev)
        prev = d


def test_place_depths_and_order():
    doc = "One two. Three four.\n\nFive six. Seven eight. Nine ten.\n\nEleven twelve."
    text, reached = place(doc, [(0.0, "START."), (1.0, "END."), (0.5, "MID.")])
    assert text.startswith("START.\n\n") and text.endswith("\n\nEND.")
    assert "\n\nMID.\n\n" in text
    assert reached[0] == 0.0 and reached[1] == 1.0 and abs(reached[2] - 0.5) < 0.2
    assert text.replace("START.\n\n", "").replace("\n\nEND.", "").replace("MID.\n\n", "").replace("\n\n", " ") \
        .split() == doc.replace("\n\n", " ").split()


def test_boundaries_skip_abbreviations():
    doc = "He met Dr. Smith in St. Louis. Then he left."
    b = boundaries(doc)
    assert doc.index("Smith") not in b and doc.index("Louis") not in b and doc.index("Then") in b


def test_key_fact_reaches_each_depth(items):
    it = items[1]
    for L in ("4k", "16k"):
        for d in C.DEPTHS:
            rd = render(it, Cell("key", L, d))
            text = rd.messages[0]["content"]
            assert text.count(it["needle"]) == 1
            assert abs(rd.reached["key"] - d) < 0.03
            assert it["answer"]["last"] not in text.replace(it["needle"], "")


def test_distractors_only_in_distract_variant(items):
    it = items[2]
    plain = render(it, Cell("key", "4k", 0.5)).messages[0]["content"]
    dist = render(it, Cell("key", "4k", 0.5, "distract")).messages[0]["content"]
    absent = render(it, Cell("absent", "4k", None, "distract")).messages[0]["content"]
    for d in it["distractors"]:
        assert d["text"] not in plain and dist.count(d["text"]) == 1 and absent.count(d["text"]) == 1
    assert it["needle"] not in absent and it["answer"]["last"] not in absent
    assert len(dist) - len(plain) < 600


def test_twohop_placements(items):
    it = items[3]
    for v, (k, h) in TWOHOP_PLACEMENTS.items():
        rd = render(it, Cell("twohop", "4k", None, v))
        text = rd.messages[0]["content"]
        assert text.count(it["needle"]) == 1 and text.count(it["hop2"]["text"]) == 1
        assert abs(rd.reached["key"] - k) < 0.03 and abs(rd.reached["hop2"] - h) < 0.03
        assert (text.index(it["needle"]) < text.index(it["hop2"]["text"])) == (k <= h)
        assert rd.gold == [it["hop2"]["town"]] and rd.bridge == [it["answer"]["last"]]
    rd = render(it, Cell("hop2", "4k", None, "far_fwd"))
    assert it["hop2_question"] in rd.messages[0]["content"]


def test_closed_book_has_no_document(items):
    rd = render(items[0], Cell("closed", variant="key"))
    assert "<document>" not in rd.messages[0]["content"] and items[0]["question"] in rd.messages[0]["content"]


def test_cell_keys_roundtrip_and_stages_unique(items):
    for st, f in STAGES.items():
        cells = f(items[10])
        assert len({c.key for c in cells}) == len(cells)
        for c in cells:
            assert Cell.from_key(c.key) == c
    aa = STAGES["aa"](items[10])
    assert all(c.tag == "aa" and c.length == items[10]["aa_length"] for c in aa)


def test_longest_prompts_fit_every_window(items):
    # at a pessimistic 4.0 characters per token the 32k prompt still fits qwen3:8b's native 32,768 and num_ctx
    for it in items[:12]:
        for cell in (Cell("key", "32k", 0.5, "distract"), Cell("twohop", "32k", None, "far_fwd")):
            chars = len(render(it, cell).messages[0]["content"])
            assert chars / 4.0 + 20 + C.NUM_PREDICT <= min(C.NATIVE_CTX["qwen3:8b"], C.NUM_CTX)

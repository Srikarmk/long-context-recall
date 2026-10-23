import pytest

from lcrecall import config as C
from lcrecall.items import build_items
from lcrecall.llm import Cache, CacheMiss, Client, cache_key, request_body
from lcrecall.prompts import Cell, render
from lcrecall.runner import TokenCheckError, check_after, check_before, est_tokens, run_one

RESP = {"text": "Zelie Thorningham", "raw": "", "output_tokens": 4, "seconds": 0.1, "done_reason": "stop"}


@pytest.fixture(scope="module")
def item():
    return build_items()[0]


def test_key_depends_on_everything():
    m = [{"role": "user", "content": "hi"}]
    o = {"temperature": 0, "seed": 7}
    k = cache_key("qwen3:8b", m, o)
    assert k == cache_key("qwen3:8b", m, dict(o))
    assert k != cache_key("llama3.1:8b", m, o)
    assert k != cache_key("qwen3:8b", m, {**o, "seed": 8})
    assert k != cache_key("qwen3:8b", [{"role": "user", "content": "hi!"}], o)
    assert k != cache_key("qwen3:8b", m, o, tag="aa")


def test_think_flag_and_options():
    assert request_body("qwen3:8b", [], {})["think"] is False
    assert request_body("qwen3:30b-a3b", [], {})["think"] is True
    assert "think" not in request_body("gemma3:12b", [], {})
    o = C.options("qwen3:8b")
    assert o["temperature"] == 0 and o["seed"] == C.SEED and o["num_ctx"] == C.NUM_CTX
    assert C.options("qwen3:30b-a3b")["num_predict"] == C.THINKING_NUM_PREDICT


def test_cache_roundtrip_and_no_network_on_hit(tmp_path):
    cache = Cache(str(tmp_path / "c.sqlite"))
    m, o = [{"role": "user", "content": "x"}], {"temperature": 0}
    cache.put(cache_key("qwen3:8b", m, o), "qwen3:8b", {**RESP, "prompt_tokens": 3})
    client = Client(cache, url="http://127.0.0.1:9", cache_only=True)
    r = client.chat("qwen3:8b", m, o)
    assert r["text"] == "Zelie Thorningham" and r["cached"] and client.stats["hits"] == 1
    with pytest.raises(CacheMiss):
        client.chat("qwen3:8b", m, o, tag="aa")
    assert len(Cache(str(tmp_path / "c.sqlite"))) == 1


def _put(cache, item, cell, model, tokens, text="Zelie Thorningham"):
    rd = render(item, cell)
    cache.put(cache_key(model, rd.messages, C.options(model), tag=cell.tag), model,
              {**RESP, "text": text, "prompt_tokens": tokens})
    return sum(len(m["content"]) for m in rd.messages)


def test_aa_repeat_has_its_own_entry_and_rows_are_checked(tmp_path, item):
    cache = Cache(str(tmp_path / "c.sqlite"))
    client = Client(cache, url="http://127.0.0.1:9", cache_only=True)
    cell = Cell("key", "1k", 0.5)
    chars = sum(len(m["content"]) for m in render(item, cell).messages)
    _put(cache, item, cell, "qwen3:8b", round(est_tokens("qwen3:8b", chars)))
    row = run_one(client, "qwen3:8b", item, cell, "short")
    assert row["outcome"] == "correct" and row["correct"] and row["cell"] == "key|1k|0.50|plain|"
    with pytest.raises(CacheMiss):
        run_one(client, "qwen3:8b", item, Cell("key", "1k", 0.5, tag="aa"), "aa")


def test_truncated_prompt_fails_loudly(tmp_path, item):
    cache = Cache(str(tmp_path / "c.sqlite"))
    client = Client(cache, url="http://127.0.0.1:9", cache_only=True)
    cell = Cell("key", "8k", 0.0)
    chars = sum(len(m["content"]) for m in render(item, cell).messages)
    # what Ollama does with an over-long prompt: keeps about half of num_ctx and says nothing
    _put(cache, item, cell, "llama3.1:8b", int(est_tokens("llama3.1:8b", chars) * 0.5))
    with pytest.raises(TokenCheckError):
        run_one(client, "llama3.1:8b", item, cell, "short")


def test_token_checks():
    o = C.options("qwen3:8b")
    check_after("qwen3:8b", 43000, round(est_tokens("qwen3:8b", 43000)), o)
    with pytest.raises(TokenCheckError):
        check_after("qwen3:8b", 43000, round(est_tokens("qwen3:8b", 43000) * 0.5), o)
    with pytest.raises(TokenCheckError):
        check_after("qwen3:8b", 43000, round(est_tokens("qwen3:8b", 43000) * 1.6), o)
    with pytest.raises(TokenCheckError):      # fits the estimate band but not num_ctx
        check_after("qwen3:8b", 150000, C.NUM_CTX, o)
    with pytest.raises(TokenCheckError):      # would not fit, refused before sending
        check_before("qwen3:8b", 200000, o)
    check_before("qwen3:8b", C.doc_chars("32k") + 600, o)

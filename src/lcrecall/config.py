"""Fixed settings for every run. Changing any of these changes the cache keys or the items."""

import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
ITEMS = os.path.join(DATA, "items.jsonl")
CORPUS = os.path.join(DATA, "corpus.jsonl")
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
FIGURES = os.path.join(REPORTS, "figures")
HANDCHECK = os.path.join(ROOT, "handcheck")
CACHE_DB = os.path.join(ROOT, "cache", "llm.sqlite")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODELS = ["qwen3:8b", "llama3.1:8b", "gemma3:12b", "qwen3:30b-a3b"]
MAIN_MODELS = MODELS[:3]

# Context windows. `ollama show` reports 40,960 for qwen3:8b, 131,072 for llama3.1:8b and gemma3:12b and
# 262,144 for qwen3:30b-a3b. The Qwen3-8B model card gives 32,768 tokens as the native window (the extra
# 8,192 in the GGUF is room for output), so the longest level is sized to stay under 32,768 for every model.
NATIVE_CTX = {"qwen3:8b": 32768, "llama3.1:8b": 131072, "gemma3:12b": 131072, "qwen3:30b-a3b": 262144}

SEED = 7
# One num_ctx for every call of a model. Ollama reloads the model whenever num_ctx changes (and drops its
# prompt cache), and a num_ctx that grew with the document would confound length with the server's
# allocation. 33,792 fits the longest prompt of any level plus the answer budget for every model; every
# call is checked against it before it is sent and after it returns (runner.check_tokens).
NUM_CTX = 33792
NUM_PREDICT = 48                  # answers are a name or a town
BASE_OPTIONS = {"temperature": 0, "seed": SEED, "num_ctx": NUM_CTX}

# The installed qwen3:30b-a3b is the thinking-only release; `think: false` does not stop it reasoning
# (found in memory-rot). It runs with thinking on and a larger cap; only the final answer is checked.
THINKING_ONLY = {"qwen3:30b-a3b"}
THINKING_NUM_PREDICT = 2048

# Document lengths. A level's document is cut from the item's filler stream at the last paragraph
# boundary within nominal x DOC_CHARS_PER_TOKEN characters. WikiText runs at 4.0 to 4.7 characters per
# token for these tokenizers, so 3.9 keeps every model's prompt at or under the nominal size (the
# measured prompt tokens are reported per model and level).
LENGTHS = {"1k": 1000, "4k": 4000, "8k": 8000, "16k": 16000, "32k": 32000}
DOC_CHARS_PER_TOKEN = 3.9
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]     # where the key fact goes: start, 25%, middle, 75%, end
BASE_LENGTH = "4k"                       # the fact-absent calls and the first two-hop stage run at this length

# Token check. Estimated prompt tokens = characters / TOKEN_CPT[model] + TEMPLATE_TOKENS[model] (median
# characters per token on the first pilot's prompts).
# A call fails loudly if the server's prompt_eval_count is below TOKEN_LOW x the estimate (Ollama cuts an
# over-long prompt to about half of num_ctx without an error), above TOKEN_HIGH x the estimate, or if
# prompt + answer budget exceeds num_ctx or the model's native window.
TOKEN_CPT = {"qwen3:8b": 4.32, "llama3.1:8b": 4.43, "gemma3:12b": 4.35, "qwen3:30b-a3b": 4.32}
# chat-template tokens around the message (role markers; qwen3:8b also gets an empty think block)
TEMPLATE_TOKENS = {"qwen3:8b": 15, "llama3.1:8b": 9, "gemma3:12b": 10, "qwen3:30b-a3b": 11}
# Per-item characters per token ranged from 3.3 to 5.1 on 1k prompts in the pilot (narrower on long ones);
# a truncated prompt comes back at about half of num_ctx or less, far outside this band.
TOKEN_LOW, TOKEN_HIGH = 0.7, 1.4

# Pinned Hugging Face source: (repo, revision, file path inside the repo, sha256 of the file).
SOURCES = {
    "wikitext2_train": ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
                        "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                        "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"),
}

N_PILOT = 6
N_MAIN = 60

# The planned grid: stage -> first N main items, sized from the pilot's seconds per call to about 8.5
# hours. Items are taken in their seeded order, so each subset is a random draw. Stages run in this order,
# each for every main model before the next starts, so a cut keeps the short and middle lengths across all
# depths first. long32 (key fact at 32k, alone and with distractors) is not in the planned grid: at about
# 23 minutes per item for the three main models it does not fit; it can be run afterwards with
# `lcrecall run --stage long32 --n-items N` and is exploratory.
GRID = {"closed": 40, "short": 40, "distract": 40, "twohop": 40, "distract_len": 40, "aa": 40, "long16": 16,
        "twohop16": 20}
# qwen3:30b-a3b (thinking) runs a reduced grid after every stage of the main models: the closed-book
# control and the two 4k stages where the main models are off ceiling.
THINK_GRID = {"closed": 20, "distract": 20, "twohop": 20}

BOOT = 2000


def options(model: str) -> dict:
    if model in THINKING_ONLY:
        return {**BASE_OPTIONS, "num_predict": THINKING_NUM_PREDICT}
    return {**BASE_OPTIONS, "num_predict": NUM_PREDICT}


def doc_chars(length: str) -> int:
    return int(LENGTHS[length] * DOC_CHARS_PER_TOKEN)

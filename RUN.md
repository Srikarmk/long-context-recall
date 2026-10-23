# Running the full grid

The full run has not been started. It needs mains power: on battery the machine may slow down or sleep
(prompt-brittleness lost five hours that way), so the start command below checks for AC power and keeps
the machine awake with `caffeinate`.

## Before starting

```sh
pmset -g batt                     # must say "Now drawing from 'AC Power'"
curl -s localhost:11434/api/tags  # Ollama is up, with qwen3:8b, llama3.1:8b, gemma3:12b, qwen3:30b-a3b
ollama ps                         # nothing loaded or "Stopping..."; if something is stuck, quit and reopen Ollama
git log --oneline -- HYPOTHESES.md   # the pre-registration is committed before any full-run call
uv sync && uv run pytest -q
uv run lcrecall plan              # 6,140 calls, about 8.5 h at the pilot's speed
```

Nothing else should be using the GPU during the run. In the pilot, one call stalled on the server for
30 minutes after an earlier client had been killed mid-request; starting from a freshly opened Ollama
avoids that, and the client now gives up on a call after 15 minutes and retries it.

## Start

From the repository root:

```sh
mkdir -p logs
pmset -g batt | grep -q "AC Power" && caffeinate -dims \
  uv run lcrecall run --grid --models qwen3:8b,llama3.1:8b,gemma3:12b --late-models qwen3:30b-a3b \
  2>&1 | tee -a logs/full.log
```

`--grid` runs the planned grid in `src/lcrecall/config.py` (`GRID` for the three main models,
`THINK_GRID` for `qwen3:30b-a3b`). Stages run in order (closed, short, distract, twohop, distract_len,
aa, long16, twohop16), each for `qwen3:8b`, `llama3.1:8b` and `gemma3:12b` one after another before the
next stage starts; then `qwen3:30b-a3b` runs closed, distract and twohop on 20 items. One call at a
time. Every call is checked against the token estimate and `num_ctx`; a mismatch (a silently truncated
prompt) stops the run with `TokenCheckError`, which means the setup is wrong and should be looked at
before resuming.

## Expected duration

6,140 calls. From the pilot's mean seconds per call by model and stage (`uv run lcrecall plan`):

| model | calls | hours | of which 16k stages |
|---|---|---|---|
| qwen3:8b | 1,960 | about 2.2 | 1.0 |
| llama3.1:8b | 1,960 | about 2.1 | 0.9 |
| gemma3:12b | 1,960 | about 3.6 | 1.4 |
| qwen3:30b-a3b | 260 | about 0.5 | 0 |

Total: about 8.5 hours, plausibly 7 to 10. Stages 1 to 6 (everything up to 8k, with the A/A floor)
take about 4.7 hours; the 16k stages about 3.3; the thinking model about 0.5. Part of the pilot ran on
battery, so mains power may be a little faster. Progress lines (`[model stage] done/total ...
elapsed=`) are printed every 25 calls.

## If it stops

Every call is cached in `cache/llm.sqlite` and every row is appended to `results/rows/<model>.jsonl` as
it finishes. Re-running the same command resumes: cached calls return at once and only missing calls go
to the server. To stop early, interrupt it; whatever finished is analysed (each comparison uses the
items answered in both of its cells), and the cut is recorded as a dated addendum in `HYPOTHESES.md`.
If progress lines stop for more than 15 minutes, check `ollama ps`; interrupting and re-running is safe.

## After the run

```sh
uv run lcrecall handcheck sample   # handcheck/sample.csv, label column left empty for Srikar
uv run lcrecall report             # reports/results.md and reports/figures/
```

Then fill in the Findings section of the README from `reports/results.md`. Once Srikar has labelled the
sample (copy `handcheck/sample.csv` to `handcheck/labels.csv` and fill `label`), `uv run lcrecall
handcheck score` writes `handcheck/agreement.json` and the next `report` includes it.

## Optional: 32k

Not in the planned grid (about 23 minutes per item for the three main models). If time allows after the
full run, `uv run lcrecall run --stage long32 --n-items 8` adds the key fact at 32k, alone and with
distractors, for the first 8 items (about 3 hours); it is exploratory and goes in an addendum.

## The pilot, for reference

The second pilot (the one reported in `reports/pilot.md`) ran:

```sh
R="uv run lcrecall run --split pilot"
$R --models qwen3:8b,llama3.1:8b --stage closed,short,distract,twohop,distract_len,aa
$R --models qwen3:8b,llama3.1:8b --stage long16,twohop16 --n-items 3
$R --models gemma3:12b --stage closed,short,distract,twohop,distract_len,aa --n-items 2
$R --models gemma3:12b --stage long16,twohop16 --n-items 1
$R --models qwen3:30b-a3b --stage closed,distract,twohop --n-items 3
$R --models qwen3:30b-a3b --stage short,distract_len,long16 --n-items 1
$R --models qwen3:8b,llama3.1:8b,gemma3:12b --stage long32 --n-items 1
```

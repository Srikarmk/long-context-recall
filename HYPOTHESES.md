# Hypotheses and analysis plan

Written 2026-10-23 at 16:02 EDT, before the full run. Nothing below is edited after the full run
starts; corrections go in a dated addendum at the end. The commit that adds this file precedes every
full-run call, so the git history shows the order.

What had been seen when this was written: two pilots on the 6 pilot items only, excluded from every
reported number. The first (prompt asking for "a name of a few words") ran `qwen3:8b` and
`llama3.1:8b` on the fact-alone grid at 1k to 16k (32k on 1 to 2 items), the 4k distractor and two-hop
stages and an A/A repeat; it led to the prompt change below, to adding distractors at every length
(the fact alone was at ceiling) and to calibrating the token check, and its calls are not used. The
second (the final prompt) ran every stage on `qwen3:8b` and `llama3.1:8b` (16k on 3 items, 32k on 1),
`gemma3:12b` on 2 items (16k and 32k on 1) and `qwen3:30b-a3b` (thinking) on the 4k distractor and
two-hop stages for 3 items and the rest for 1: 804 calls. The pilot was used to check the templates,
the checker, the token check, timing and the grid size. On far too few items to estimate anything, it
showed:

- every call passed the token check: the server's prompt_eval_count was 0.93 to 1.07 times the
  estimate; the longest prompt was 27,642 tokens (`qwen3:8b`, 32k level), under `num_ctx` 33,792 and
  the 32,768 native window; no answer hit its length cap (48 tokens; 2,048 with thinking);
- every reply was a short answer; the checker read 799 of 804 as one of correct, near miss or
  abstain; the other 5 were `gemma3:12b` guessing a name closed-book (4, all wrong) and one reply that
  listed all four candidate people (`mixed`);
- closed-book: no correct answer from any model; `qwen3:8b`, `llama3.1:8b` and `qwen3:30b-a3b` said
  "unknown" every time, `gemma3:12b` guessed;
- the fact alone: 100% at every length (1k to 32k) and depth for `qwen3:8b` and `gemma3:12b`;
  `llama3.1:8b` 93 to 100% per length, its misses being "not stated" with the fact at 75% or the end;
- with three near-miss distractors, recall over all depths at 1k / 4k / 8k / 16k was 67 / 77 / 67 / 40%
  (`qwen3:8b`), 57 / 37 / 20 / 20% (`llama3.1:8b`) and 90 / 60 / 40 / 40% (`gemma3:12b`, 1 or 2
  items); wrong replies named a distractor's person (113 calls), never an invented one; no consistent
  middle dip; which distractor was named varied by model and length;
- with the fact left out, every model said "not stated" without distractors; with them, `qwen3:8b`
  named a distractor on 2 of 6 items, `llama3.1:8b` on 3 of 6, `gemma3:12b` on 2 of 2;
- two-hop at 4k: adjacent facts 100% for every model; with the facts at 10% and 90%, `qwen3:8b` and
  `gemma3:12b` failed only when the birthplace came first (50% and 1 of 2, naming a decoy town),
  `llama3.1:8b` only when the person's link came first (67%); at 16k `llama3.1:8b` replied "not
  stated" on all 6 far calls;
- in the first pilot, asking for "a name" made `llama3.1:8b` answer the two-hop town question with
  the person's name on 7 of 12 far calls; with "the answer, in a few words" this fell to 0 of 12;
- the A/A repeat (30 calls each for the 8B models, 10 for `gemma3:12b`) changed no answer;
- `qwen3:30b-a3b` with thinking answered every call with the fact present correctly (3 items at 4k, 1
  elsewhere), but with the fact left out and distractors present it named a distractor on 2 of 3 items
  (about 300 output tokens per call);
- median seconds per call (with prefix reuse within an item), 1k / 4k / 8k / 16k / 32k: `qwen3:8b`
  0.9 / 3.0 / 6.8 / 16.7 / 41.6, `llama3.1:8b` 0.8 / 2.8 / 4.4 / 15.5 / 43.2, `gemma3:12b`
  1.9 / 6.2 / 7.4 / 22.7 / 44.6, `qwen3:30b-a3b` 4.2 / 6.5 / 8.2 / 16.0 (no 32k). One call stalled
  for 30 minutes on the server after an earlier client was killed mid-request; the client now times out
  after 15 minutes and retries. Part of the second pilot ran on battery.

Predictions informed by the pilot are marked (P).

## Primary question

When a model must answer from a long document a question about a fact it cannot already know, how
often does it find and use the fact, and how does that change with the document's length, with the
fact's position in it, and with near-miss facts about similar entities nearby? When the answer needs
two facts, does it matter how far apart they are and in which order?

## Definitions

- *Recall* (accuracy) of a cell: share of items whose reply is `correct` (names the right answer and
  no near miss). With the fact absent, the right reply is to abstain.
- *Fact alone* / *with distractors*: the key fact with no other planted fact, or with the three
  near-miss facts at their seeded depths.
- *Length effect*: recall over all five depths at a length minus at 1k, paired by item.
- *Middle minus ends*: recall at 50% minus the mean of recall at 0 and 100%, paired by item; pooled over
  lengths by averaging an item's lengths.
- *Distractor effect*: recall with distractors minus without, same items, lengths and depths.
- *Before share*: of `wrong_entity` replies that name exactly one distractor, the share naming one that
  sits before the key fact; *chance*: the share expected if the reply named one of the three distractors
  at random (the mean, over those calls, of the fraction of distractors before the key fact).
- *Flip rate* (A/A floor): share of repeated calls whose correctness differs from the first call.
- *Floor rule*: an effect counts only when its 95% interval excludes zero and its size exceeds the
  model's flip rate.
- "Main models" = `qwen3:8b`, `llama3.1:8b`, `gemma3:12b`.

## Predictions

1. No item is answered correctly closed-book by any model.
2. (P) With the fact alone, recall over all depths is at least 90% at every length from 1k to 16k for
   every main model.
3. (P) With the fact alone, length barely matters: for every main model, 16k minus 1k has the lower end
   of its interval at or above -5 pp.
4. (P) At 4k, the near-miss distractors lower recall for every main model (interval below zero, beyond
   the floor).
5. With distractors, pooled over 1k, 4k and 8k, recall with the fact in the middle is below the mean of
   start and end (interval below zero, beyond the floor) for at least two main models (the
   lost-in-the-middle shape; the pilot did not show it).
6. (P) With distractors, recall at 16k is below recall at 1k (interval below zero, beyond the floor) for
   at least two main models.
7. When a reply names a distractor, it names one placed before the key fact more often than chance, by
   more than 10 pp, for at least two main models (in the pilot only `llama3.1:8b` did).
8. (P) With the fact left out, every main model abstains on at least 90% of items when there are no
   distractors, and abstains less with distractors (interval below zero) for at least two.
9. (P) Two-hop at 4k: adjacent facts are answered on at least 90% of items by every main model; at
   least one main model is less accurate with the facts far apart (either order; interval below zero,
   beyond the floor); and at least one is less accurate on the far-forward two-hop question than on its
   second hop alone in the same document (interval below zero, beyond the floor).
10. (P) Two-hop with the facts far apart (both orders pooled) is less accurate at 16k than at 4k
    (interval below zero, beyond the floor) for at least two main models.
11. (P) The A/A flip rate is under 3% for every model.
12. (P) `qwen3:30b-a3b` (thinking) is more accurate than `qwen3:8b` with distractors at 4k on the same
    items (interval above zero). Confounded with size and thinking.
13. Two-hop at 4k with the facts far apart is less accurate when the birthplace comes first and the
    entity's link to the person last than in the forward order (interval below zero) for at least two
    main models (the pilot showed it for two and the reverse for `llama3.1:8b`).

What would contradict "models lose facts in long contexts": recall near 100% in every cell with
distractors as well as without, distractor effects within the A/A floor, two-hop as accurate far apart
as adjacent, and abstention unchanged by near misses when the fact is absent.

## Analysis plan (fixed in advance)

- Unit of resampling: item. 95% percentile bootstrap over items, 2,000 draws. Every cell of an item
  holds the same fact; every comparison is paired by item and uses the items answered in both of its
  cells (relevant if the run is cut). Length and position summaries use items answered at all five
  depths.
- Primary metrics: recall by model x length x depth, alone and with distractors; length effect; middle
  minus ends, interior (25/50/75%) minus ends, start minus end; distractor effect by length and depth;
  outcome shares (near miss named, abstain, other); fact-absent outcomes; two-hop recall by placement
  and length, the second hop alone, and outcome shares (decoy town, person only, abstain); A/A flip
  rate; closed-book outcomes.
- Secondary (exploratory): which kind of distractor is named; before share and nearest-distractor
  share against chance; per-relation breakdowns; anything at 32k if it is run later.
- Floor rule as defined above, using each model's own flip rate.
- The checker's rules are fixed as in `src/lcrecall/check.py` at this commit; any later change is an
  addendum, and the report always re-checks every stored output with the current rules. A seeded
  sample of outputs goes to `handcheck/` for a labelling pass by Srikar after the run; until it is
  labelled, the checker check is reported as not done.
- Multiple comparisons: no correction. Predictions 1 to 11 are read as confirmatory; 12 and 13 and
  everything else as exploratory.
- Exclusions: pilot items. A call that fails the token check stops the run (it means the setup is
  wrong). A call that errors is retried five times, then the run stops; re-running resumes from the
  cache.

## Run settings

- Ollama 0.40.6 on an Apple M5 Pro (48 GB). Models: `qwen3:8b` (digest `500a1f067a9f`, think off),
  `llama3.1:8b` (`46e0c10c039e`), `gemma3:12b` (`b92b218582c5`), `qwen3:30b-a3b` (`ad815644918f`,
  think on; the installed tag is the thinking-only release). All Q4_K_M.
- Temperature 0, seed 7, `num_ctx` 33,792 on every call, `num_predict` 48 (2,048 for the thinking
  model; only its final answer is checked).
- Token check: estimate = characters / (4.32 `qwen3:8b` and `qwen3:30b-a3b`, 4.43 `llama3.1:8b`, 4.35
  `gemma3:12b`) + template tokens (15, 9, 10, 11); a call stops the run if the server's prompt_eval_count
  is outside 0.7 to 1.4 times the estimate, or if prompt plus answer budget exceed `num_ctx` or the
  native window (32,768 for `qwen3:8b`).
- One call at a time. Stages run in order, each for the three main models one after another, then
  `qwen3:30b-a3b`. Within a stage, an item's calls are sent together, sorted by prompt text.
- Items: 60 main items built with seed 7 (`data/items.jsonl`); the grid uses the first N in that order.

## Exact grid

| order | stage | cells per item | items | calls per main model |
|---|---|---|---|---|
| 1 | closed | key question and second-hop question, no document | 40 | 80 |
| 2 | short | fact alone, 1k / 4k / 8k x 5 depths | 40 | 600 |
| 3 | distract | with distractors, 4k x 5 depths; fact absent at 4k, without and with distractors | 40 | 280 |
| 4 | twohop | 4k: adjacent, far forward, far backward; second hop alone | 40 | 160 |
| 5 | distract_len | with distractors, 1k and 8k x 5 depths | 40 | 400 |
| 6 | aa | the 5 with-distractor cells at the item's A/A length (1k, 4k or 8k in rotation), again | 40 | 200 |
| 7 | long16 | 16k x 5 depths, alone and with distractors | 16 | 160 |
| 8 | twohop16 | the four two-hop calls at 16k | 20 | 80 |

`qwen3:30b-a3b` (thinking), after all of the above: closed, distract and twohop on the first 20 items
(260 calls). 32k is not in the grid.

In all, 6,140 calls. Expected duration from the second pilot's mean seconds per call by model and
stage (`lcrecall plan`): `qwen3:8b` 2.2 h, `llama3.1:8b` 2.1 h, `gemma3:12b` 3.6 h, `qwen3:30b-a3b`
0.5 h, about 8.5 hours in all (7 to 10 given the spread of the pilot timings and part of the pilot
having run on battery). If the run is cut, whatever finished is reported and the analysed grid is
stated in an addendum.

## Addenda

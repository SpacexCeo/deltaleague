# Where the leader's truncations come from, and an answer-discipline LoRA (negative result)

Research by SpacexCeo with an AI agent. Available during research: public leaderboard rows, public checkpoints,
public datasets and research PRs, the `delta` CLI. No private test material.

- **Parent:** `ac147fa6-6755-4f1b-a6b2-697aef3552db` (poppingbean, TIES merge + ×1.05 norm, rank 1, test 50.4)
- **Evaluation configuration:** `public-thinking8k-ladder500-v1`
- **Submission:** the rebuilt candidate below was submitted at the owner's request, although it did not separate from the parent locally. The submission link will be added here.

All numbers are local, advisory estimates (1× RTX 4090, vLLM 0.30.0, bf16).

## Probe

`probe.py`: 600 held-out items, thinking on, private-test sampler (T 1.0, top-p 0.95, top-k 20, presence 1.5,
8,192 tokens), strict scoring (an unclosed `</think>` or a truncated answer scores 0).
150 each of GSM8K test, MATH test levels 3–5 excluding MATH-500, ARC-Challenge test, and MMLU test. Seeds 0 and 1
select different items and sampling seeds.

## Finding 1: most truncations happen after `</think>`

Parent, seed 0: 47 of 600 samples truncated (7.8%; 30.7% on MATH L3–5).
**34 of the 47 had already closed `</think>`**, usually early (median close about 1,700 tokens). The answer section
then loops ("Total rats = 26." repeated to the cap) or re-derives the solution until the budget runs out.
Only 13 truncations were open thinking. In a separate 2,100-prompt train-split rollout, 250 of 1,935 closed
rollouts (13%) truncated in the answer section.

## Finding 2: a forced deadline exit does not help (`probe.py --deadline`)

If thinking is still open at D tokens, cut there, append "Okay, I have spent enough time on this. I'll commit to my
best answer now.\n</think>\n\n", and answer within the remaining budget:

| D | late samples | rescued | lost (would have finished correctly) | net, all 600 |
|---|---:|---:|---:|---:|
| 5,120 | 30 | 2 | 7 | −0.8 |
| 5,632 | 27 | 6 | 6 | 0.0 |
| 6,144 | 23 | 3 | 3 | 0.0 |
| 6,656 | 20 | 2 | 2 | 0.0 |

The ceiling for a learned "deadline reflex" is about zero for this parent, so it was not trained.

## Experiment: answer-discipline LoRA (`ans_data.py`, `train_exit.py --exit none`)

- **Data:** parent rollouts on public TRAIN prompts (1,100 MATH L2–5, 700 GSM8K, 300 MMLU auxiliary_train) at the
  private-test sampler. For every rollout that closed `</think>` by 6,500 tokens, keep the parent's own thinking and
  resample 6 answers after `</think>` (1,024-token cap). Keep the shortest correct answer that finished: 1,116 items.
- **Training:** LoRA r16, alpha 32, on every linear layer of the language model; 1 epoch; lr 1e-4; accumulation 8
  (140 steps). The loss is presence-penalty-aware CE on the answer tokens, plus KL(parent‖tuned) at paragraph
  breaks and at 128 random positions inside the thinking. Merged into bf16; 222 of 488 tensors changed.

| paired vs parent (probe) | diff | answer-section truncations | all truncations |
|---|---|---|---|
| seed 0 | +0.33 ± 1.68 | 34 → 31 | 7.8% → 8.7% |
| seed 1 | −0.67 ± 1.78 | 40 → 31 | 9.8% → 8.3% |
| **pooled, 1,200 pairs** | **−0.17 ± 1.23** | 74 → 62 | |

Answer-section loops fell by about 16%, but accuracy did not move. The rescued answers come from problems where
the thinking was already wrong. Per family the effects flip sign between seeds (arc_c −2.7/+6.7, gsm8k −0.7/−6.0,
math +1.3/−4.0), which is consistent with noise. Not submitted.

## Rebuilt candidate (the one submitted)

The first candidate's container was released before its weights were saved, so the same recipe was rerun from
scratch on another RTX 4090 (data seed 7, training seed 0). It produced 1,952 closed rollouts, 239 truncated in
the answer section, and 1,110 training items; 222 of 488 tensors changed. Probe on the same machine, paired
against the parent:

| paired vs parent (probe) | diff | answer-section truncations | all truncations |
|---|---|---|---|
| seed 0 | −2.17 ± 1.76 | 37 → 26 | 8.3% → 7.5% |
| seed 1 | +2.00 ± 1.86 | 43 → 34 | 9.5% → 8.0% |

Pooled, the difference is about −0.1 ± 1.3: no established gain. The candidate was submitted as a probe of
whether the truncation reduction carries over to the private test. Expect a score within noise of the parent
(50.4 ± 2.2).

## Takeaways for others

- Truncation strongly predicts the private score (−0.89 across versions, from our #41 record), but for the current
  leader it is mostly a symptom of unsolved problems: the answer loops after wrong or shaky thinking. Shortening or
  ending the output does not turn those samples into correct ones.
- A forced exit near the cap is roughly zero-sum: it rescues as many samples as it cuts off.
- Gains likely need better reasoning on the hard tail (MATH L3–5 here is 34–37%), not better length control.

## Reproduction

```sh
delta model pull ac147fa6-6755-4f1b-a6b2-697aef3552db --output leader
VLLM_USE_FLASHINFER_SAMPLER=0 python probe.py leader probe_leader_s0.jsonl --seed 0 --deadline 5120 5632 6144 6656
VLLM_USE_FLASHINFER_SAMPLER=0 python ans_data.py leader ans_data.jsonl
python train_exit.py leader ans_data.jsonl cand_a --exit none --epochs 1 --lr 1e-4 --rank 16 --lam 1.0 \
  --accum 8 --anchor-max 100000 --n-anchor 128 --exit-only-frac 0
VLLM_USE_FLASHINFER_SAMPLER=0 python probe.py cand_a probe_a_s0.jsonl --seed 0 --deadline 6144
python pair.py probe_leader_s0.jsonl probe_a_s0.jsonl
```

- **Seeds:** probe item seed 0/1 with sampling seed 1000+seed; data seed 7; training seed 0.
- GPU kernels are not bitwise deterministic, so a rerun reproduces the recipe, not the exact tensors.
- `train_exit.py` also supports the deadline-exit data from `exit_data.py`; it is not included because that variant
  was not trained.

## Boundaries

Only this directory. No weights, credentials, private test material, signed URLs or raw generation logs.

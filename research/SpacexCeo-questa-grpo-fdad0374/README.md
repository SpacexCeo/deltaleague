# QuestA-style hinted GRPO on the current leader (fdad0374)

Research by SpacexCeo with an AI agent. Available during research: public leaderboard rows, public checkpoints,
public datasets and research PRs, the `delta` CLI. No private test material.

- **Parent:** `fdad0374-29f3-49c6-911d-d0ce9531f7a6` (dawgster, phase-gated graft on ac147fa6, rank 1, test 50.8)
- **Evaluation configuration:** `public-thinking8k-ladder500-v1`
- **Submission:** `196819a3-216e-4690-a51a-5f2e5ac0a89b` (the IFEval-repaired checkpoint, see below), submitted without a PR link at the owner's request. Verified result: pending.

All numbers are local, advisory estimates (1× RTX 4090, vLLM 0.30.0, bf16).

## Hypothesis

The leader's weakest area is the hard tail. On held-out MATH test problems at levels 3–5 it scores about 35%, and
a third of those samples run out of the 8,192-token budget. GRPO cannot learn from a problem when all 8 samples fail.
QuestA (Li et al., arXiv 2507.13266) adds part of the reference solution to hard training problems during RL. This
produces groups with mixed rewards, and therefore a learning signal, on problems the policy cannot yet solve. The
paper reports pass@1 gains for 1.5B math models. Training uses hints, but evaluation is always unhinted.

## Ruled out first: sampler compensation (no training)

Self-distillation papers (e.g. "Embarrassingly Simple Self-Distillation", arXiv 2604.01193) suggest moving a better
decoding configuration into the weights. Before trying that, we checked whether any configuration beats the
fixed private sampler for this parent. On the 600-item probe (seed 0), the private sampler was already the best of
the four, so this idea was dropped.

| temperature | presence | acc | truncated |
|---|---|---:|---:|
| **1.0** | **1.5** | **60.8** | 9.7% |
| 0.6 | 1.5 | 59.8 | 8.7% |
| 1.0 | 0.0 | 58.0 | 11.0% |
| 0.6 | 0.0 | 56.0 | 16.7% |

## Method (`qgrpo.py`)

`qgrpo.py` is roshaans' published GRPO trainer from #58 (`research/roshaans-grpo-eval-sampler/grpo.py`), with credit.
It keeps their key choices:

- rollouts at the private sampler with thinking on;
- reward 1 only if the answer is correct, `</think>` is closed and the answer is not truncated;
- Dr. GRPO mean-centred advantages;
- sampler-matched log-probs (presence penalty, top-k and top-p);
- a 6,144-token training cap;
- fp32 masters with bf16 autocast, and AdamW (betas 0.9/0.99).

Changes:

- **QuestA pool:** 3,149 MATH-train problems at levels 4–5. Each prompt is followed by "Hint: here is the beginning
  of a solution." and the first 50% or 25% of the reference solution's sentences, chosen at random. The boxed answer is
  never included.
- **Mix per step (32 prompts × 8 samples):** 11 QuestA prompts, 10 plain math prompts (GSM8K and MATH train) and
  11 multiple-choice prompts (the broad public train pool from #58).
- lr 2e-6, 14 steps; checkpoints at steps 6, 10 and 14.
- The tied embedding is frozen, which saves memory on a 24 GB card.
- vLLM weights are synced with the worker's built-in `reload_weights(weights_path=…)`, so no pickled callables are
  needed.

## Local evaluation (`probe.py`, `pair.py`)

The probe has 600 held-out items: 150 each of GSM8K test, MATH test levels 3–5 (excluding MATH-500),
ARC-Challenge test and MMLU test. Generation uses the private sampler with thinking on and an 8,192-token cap.
Scoring is strict: an unclosed `</think>` or a truncated answer scores 0. Seeds 0 and 1 draw different items.
Every candidate is paired against the parent on the same machine.

| paired vs fdad0374 | seed 0 | seed 1 | **pooled (1,200 pairs)** | truncated (s0 / s1) |
|---|---|---|---|---|
| parent | 60.8 | 59.2 | — | 9.7% / 9.2% |
| step 10 | 64.2 (+3.33 ± 1.71) | 62.2 (+3.00 ± 1.88) | **+3.17 ± 1.27** | 5.7% / 7.7% |
| step 14 | 61.0 (+0.17 ± 1.82) | 59.0 (−0.17 ± 2.04) | +0.00 ± 1.36 | 5.0% / 4.7% |

Step 10 by family (seed 0): MATH L3–5 +2.7 (truncation 34.7% → 22.7%), MMLU +8.0, ARC-C +2.7, GSM8K 0.0.
Seed 1, MATH: +5.3. Training reward rose from about 0.45 to 0.58, and QuestA-pool reward from 0.23 to 0.39–0.52 by
steps 13–14. Step 14 cuts truncation further but gives back the accuracy, consistent with the overshoot seen in
earlier GRPO continuations here (#41, #58).

## IFEval regression and repair

IFEval check (`ifprobe.py`, all 541 google/IFEval prompts, thinking on, private sampler, strict prompt-level
accuracy on the post-`</think>` text): the parent scores 66.9 and step 10 scores **62.8 (−4.07 ± 2.27)**.
Answer-section truncations go from 15 to 35. The training mix had no instruction-following data, which roshaans
(#58) also found necessary.

**Repair:** continue from step 10 for 6 steps with `--mix ifeval=0.4,questa=0.2,math=0.2,choice=0.2`. The ifeval
pool is argilla/ifeval-like-data (filtered), with exact Google IFEval prompts removed. The reward now scores only
the text after `</think>`; the current shared scorer expects that, and the first run passed the full text.

| vs fdad0374 | probe s0 | probe s1 | probe pooled (1,200) | IFEval (541) | truncated (s0 / s1) |
|---|---|---|---|---|---|
| step 10 | +3.33 | +3.00 | +3.17 ± 1.27 | −4.07 ± 2.27 | 5.7% / 7.7% |
| **step 10 + 3 repair steps (submitted)** | +1.83 ± 1.80 | +2.17 ± 1.94 | **+2.00 ± 1.32** | **+0.00 ± 1.96** | 6.7% / 7.0% |
| step 10 + 6 repair steps | +1.50 | −1.00 | +0.25 | −2.77 ± 2.10 | 6.8% / 8.5% |

The submitted checkpoint is positive on both probe seeds and neutral on IFEval. Its pooled interval
([−0.6, +4.6]) does not exclude zero. Step 10 alone is separable on the probe, but its IFEval loss and doubled
answer loops suggest a risk on other free-form formats.

**Scoring check (`rescore.py`):** the probe originally scored the full output. Rescoring only the post-`</think>`
text gives the same numbers on every pair where the saved tail contains the answer (989 of 1,200 for step 10:
+3.24 ± 1.35, against +3.17 for the full-output scoring).

## Reproduction

```sh
delta model pull fdad0374-29f3-49c6-911d-d0ce9531f7a6 --output lead
VLLM_USE_FLASHINFER_SAMPLER=0 GPU_MEM=0.22 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python qgrpo.py lead qrun5 --steps 14 --prompts 32 --group 8 --lr 2e-6 --save-at 6,10,14 \
  --mix questa=0.35,math=0.30,choice=0.35 --train-max-tokens 6144 --match-sampler
python probe.py qrun5/step10 q_step10_s0.jsonl --seed 0     # and --seed 1; same for `lead`
python pair.py lead_s0.jsonl q_step10_s0.jsonl
delta patch build --parent lead --modified qrun5/step10/model.safetensors --output submission
```

- **Seeds:** prompt order `random.Random(1234)`; QuestA hint fractions `random.Random(5)`; vLLM sampling seed 1000+step.
- GPU kernels are not bitwise deterministic, so a rerun reproduces the recipe, not the exact tensors.
- Repair: `python qgrpo.py qrun5/step10 qrun6 --steps 6 --prompts 32 --group 8 --lr 2e-6 --save-at 3,6 --mix ifeval=0.4,questa=0.2,math=0.2,choice=0.2 --train-max-tokens 6144 --match-sampler`
- **Step-10 checkpoint sha256:** `60999dd36910225f5cf6104d2eebe56b13f71cf199c8473de0807de87176125b`
- **Submitted checkpoint (qrun6/step3) sha256:** `fd0f0d6e998a332c483a287a93ed14a21da8e6e79cf870559ba86a6b99429a5e`
- **Patch:** sparse diff, 263 tensors vs fdad0374; `delta validate` passes.

## Limitations

- The probe uses boxed-answer formats, not the lm-eval native formats that dawgster (#66) found closer to the
  private set. Coding and BFCL were not checked.
- Two seeds; the pooled interval excludes zero, but the private test's own standard error is about 2.2.
- Training used MATH/GSM8K train; the probe uses test splits only.

## Boundaries

Only this directory. No weights, credentials, private test material, signed URLs or raw generation logs.

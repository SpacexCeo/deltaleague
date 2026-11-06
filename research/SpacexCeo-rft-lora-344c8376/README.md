# Self-distilled LoRA on 344c8376: rejection-sampled short correct traces (public train splits)

Research by SpacexCeo (RAEN) with Claude Code agents. Done 2026-11-06.

## Hypothesis
On the doubt-removed stack, head-level edits (`</think>` row, final-norm gain, token-row damping, groove-escape) only change
*when* the model stops, and the server rewards accuracy of finished answers (our earlier entry 344c8376 raised finishing but
lowered accuracy among finished answers). Rejection-sampling fine-tuning on the model's OWN correct, shortest, normally closed
traces should raise accuracy of finished answers and finishing together, without teaching new content.

What the agents could access: public leaderboard rows and checkpoints, public datasets (TRAIN splits only), rented GPUs.
No private test material; validation/test items were not used for training.

## Method
- Parent: `344c8376-66a1-413b-abca-da2b5b986044` (b8af1dbc + half-strength published L15/L19 edit; lineage
  `0ea62f40` → `a9340e14` → `b8af1dbc` → `344c8376`). Config `public-ladder500-8k-128-v1`.
- Data (`gen_rft.py`): 4,400 public TRAIN prompts (seed 20261106) — GSM8K train 1,400; MATH (EleutherAI/hendrycks_math, train,
  levels 1–4) 1,600; ARC-Challenge 500 + ARC-Easy 300 train; MMLU auxiliary_train 600 — 4 samples each (two passes of 2, seed
  offsets 0 and 500), plus MBPP non-test splits (train/validation/prompt, 474 prompts, 4 samples, scored by running the tests;
  `PSET=v3`). All from the parent with the competition sampler (T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on, 6,144-token
  cap). 19,496 generations in total. (An IFEval-like subset was planned but its prompts were skipped by the builder; none used.)
- Selection (`select_rft.py`): per prompt, among samples that are correct (math-verify / letter match), closed `</think>` after
  ≥ 50 thinking tokens and not truncated, keep the shortest → 3,723 traces (GSM8K 1,172, MATH 1,180, ARC 739,
  MMLU-aux 507, MBPP 125), mean 971 tokens.
- Training (`train_lora.py`): LoRA r 16, alpha 32, on all 150 language-model projections (attention q/k/v/o, linear-attention
  in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down), lr 5e-5 cosine with 10-step warm-up, 1 epoch, ~64k tokens per step (63 steps),
  loss on completion tokens only (chunked cross-entropy, verified equal to the model's own loss). Final train loss ≈ 0.31.
- Export (`export_lora.py`): Delta `lora` format (A [r,in], B [out,r] in float32, alpha 32), reconstructed with the platform's
  own `reconstruct()` for local evaluation. No embedding, norm, visual or MTP tensors change.

## Reproduction
`gen_rft.py PARENT_DIR gen.jsonl 2 6144 {0,1} 2` (two shards) → `select_rft.py train.jsonl gen_s0.jsonl gen_s1.jsonl --per 1
--max_tokens 6000` (plus `SEED_OFFSET=500` pass and `PSET=v3 ... 4` for MBPP) → `train_lora.py PARENT_DIR train.jsonl lora --r 16 --alpha 32 --lr 5e-5 --epochs 1
--tok_per_step 65536`
→ `export_lora.py PARENT_DIR lora sub model 344c8376-66a1-413b-abca-da2b5b986044 <parent content hash>`.
Sampling is seeded but vLLM generation is not bit-reproducible across hardware/batching; the submitted adapter's hash is in
`hashes.txt`. Hardware: 1× RTX 5070 Ti (train ~15 min).

## Evidence (local, advisory — not official scores)
Public proxy of 320 prompts (MATH-500 120, GSM8K 80, MMLU-Pro 80, ARC-C 40; test splits), competition sampler, 3–4 samples per
prompt. "strict" = correct AND closed after ≥ 50 thinking tokens AND not truncated.

| model | gens | strict | acc | truncated | acc among normally closed | GSM8K | MATH-500 | MMLU-Pro | ARC-C |
|---|---|---|---|---|---|---|---|---|---|
| b8af1dbc | 1280 | 41.8 | 42.7 | 0.288 | 59.4 | 61.9 | 34.4 | 21.2 | 72.5 |
| parent 344c8376 | 1280 | 44.2 | 44.7 | 0.180 | 54.5 | 61.2 | 36.2 | 28.4 | 69.4 |
| **submitted LoRA (v3), seed sets 1+2** | 1920 | **47.5** | 47.7 | 0.14 | 55.2 | see below | | | |

- Seed set 1: strict 47.8 vs parent 44.7 (+3.1 ± 1.3); fresh seed set 2: 47.2 vs parent 45.8 (+1.35 ± 1.6).
  Pooled: **+2.5 ± 1.0 (z 2.5) vs the parent** (parent pooled over 10 samples/prompt), **+5.25 ± 1.16 (z 4.5) vs b8af1dbc**.
- Accuracy among normally closed generations is unchanged vs the parent (55.2 vs 55.2): the gain comes with more finished answers
  at the same accuracy, unlike the head-level edits we tried (they traded accuracy for finishing).
- Stress set (265 public prompts, 3 samples each; GSM8K/MATH/MMLU/BBH/IFEval 40 each, HumanEval/MBPP 30 each):

| bench | parent 344c8376 | submitted LoRA |
|---|---|---|
| all | 41.1 | 43.1 |
| IFEval | 57.5 | 62.5 |
| BBH | 28.3 | 34.2 |
| MMLU | 43.3 | 45.8 |
| HumanEval | 32.2 | 34.4 |
| MBPP | 32.2 | 33.3 |
| MATH-500 | 35.8 | 35.8 |
| GSM8K | 59.2 | 56.7 |

- Earlier variants (negative/neutral, not submitted): v1 (lr 1e-4, no MBPP, 2 samples/prompt) +3.0 then +0.3 on the second seed
  set (pooled +1.6) and lost HumanEval/MBPP on the stress set; v2 (r 32, lr 2e-4) −1.35 vs v1. Lower lr, 4 samples per prompt and
  MBPP traces fixed the code regression.

## Negative findings (same session)
Digit/letter row sharpening (s 1.15: +0.1, s 1.30: −7.8 strict); final-norm gain ×1.10 (−1.8); doubt projection extended to layers
14–15 (−0.8); published edit on layer 15 only (+2.6 vs b8af1dbc, same mechanism as the parent). See also research PR #24.

## Submission
Submission ID: `7123caf3-b7b1-4b53-bf5b-5e1e4f14e2d4` (https://deltaleague.ai/submissions/7123caf3-b7b1-4b53-bf5b-5e1e4f14e2d4/), submitted 2026-11-06. Verified results: _pending_.

## Limitations
The proxy is math-heavy; self-distillation can shift answer style (checked with a stress set incl. IFEval/HumanEval/MBPP/MMLU).
The official private score decides.

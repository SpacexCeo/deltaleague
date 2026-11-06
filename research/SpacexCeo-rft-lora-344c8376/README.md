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
- Data (`gen_rft.py`, seed 20261106): 4,400 public TRAIN prompts — GSM8K train 1,400; MATH (EleutherAI/hendrycks_math, train,
  levels 1–4) 1,600; ARC-Challenge 500 + ARC-Easy 300 train; MMLU auxiliary_train 600 — 2 samples each from the parent with the
  competition sampler (T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on, 6,144-token cap).
- Selection (`select_rft.py`): per prompt, among samples that are correct (math-verify / letter match), closed `</think>` after
  ≥ 50 thinking tokens and not truncated, keep the shortest → 3,285 traces, 3.57M completion tokens.
- Training (`train_lora.py`): LoRA r 16, alpha 32, on all 150 language-model projections (attention q/k/v/o, linear-attention
  in_proj_qkv/in_proj_z/out_proj, MLP gate/up/down), lr 1e-4 cosine, 1 epoch, ~64k tokens per step (61 steps), loss on completion
  tokens only (chunked cross-entropy, verified equal to the model's own loss). Final train loss ≈ 0.31.
- Export (`export_lora.py`): Delta `lora` format (A [r,in], B [out,r] in float32, alpha 32), reconstructed with the platform's
  own `reconstruct()` for local evaluation. No embedding, norm, visual or MTP tensors change.

## Reproduction
`gen_rft.py PARENT_DIR gen.jsonl 2 6144 {0,1} 2` (two shards) → `select_rft.py train.jsonl gen_s0.jsonl gen_s1.jsonl --per 1
--max_tokens 6000` → `train_lora.py PARENT_DIR train.jsonl lora --r 16 --alpha 32 --lr 1e-4 --epochs 1 --tok_per_step 65536`
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
| **this LoRA, seed set 1** | 960 | **47.2** | 47.5 | 0.141 | 55.0 | 67.5 | 38.1 | 29.6 | 71.7 |
CONFIRM_ROWS

Strict difference: +3.0 ± 1.4 vs the parent, +5.4 ± 1.5 (z 3.7) vs b8af1dbc. Unlike all head-level edits we tried, accuracy among
normally closed generations does not fall relative to the parent.
STRESS_ROWS

## Negative findings (same session)
Digit/letter row sharpening (s 1.15: +0.1, s 1.30: −7.8 strict); final-norm gain ×1.10 (−1.8); doubt projection extended to layers
14–15 (−0.8); published edit on layer 15 only (+2.6 vs b8af1dbc, same mechanism as the parent). See also research PR #24.

## Submission
Submission URL/ID: _added after `delta submit … --pr`_

## Limitations
The proxy is math-heavy; self-distillation can shift answer style (checked with a stress set incl. IFEval/HumanEval/MBPP/MMLU).
The official private score decides.

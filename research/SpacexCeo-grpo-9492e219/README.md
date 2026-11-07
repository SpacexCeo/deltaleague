# GRPO at the evaluation sampler, continued from 9492e219 (work in progress)

Parent: `9492e219-6f07-4bdc-b863-e86274ee62c2` (ma4ypic4y, "GRPO at the evaluation sampler, 20 steps", itself on our
`344c8376`). Credit: the idea of reward-optimising at the server sampler is theirs; no recipe was published, this is an
independent implementation.

## Why
- Across the 34 scored versions, private test correlates −0.89 with the server truncation rate and only 0.15 with the
  validation score. 9492e219 has the lowest truncation (0.116) and keeps accuracy on finished answers.
- Weight diff 9492e219 vs 344c8376: 271/488 tensors (every linear weight of all 24 layers + embeddings), relative change
  0.056%, high-rank updates — full-parameter training, not LoRA or a direction edit. `</think>` / EOS rows barely move.
- Our surgical edits on cd0b95bb (doubt / re-check / norm remixes, anti-enumeration and anti-waffle direction ablations)
  all failed to beat their parent locally: they move failures between truncation modes without adding reasoning.

## Method (`grpo.py`, `make_pool.py`)
- Rollouts: vLLM with the server sampler (T 1.0, top-k 20, top-p 0.95, presence 1.5, thinking on, 8192 tokens),
  8 samples × 32 prompts per step from public TRAIN splits only (GSM8K, MATH L1–4, ARC, CSQA, Winogrande, BoolQ, PIQA,
  OBQA, MMLU auxiliary_train), weighted toward intermediate difficulty.
- Reward: correct AND `</think>` closed AND not truncated.
- Update: on-policy, group-normalised advantages, one AdamW step per GRPO step, full parameters except the tied
  embedding, fp32 master weights with bf16 autocast; checkpoint written in bf16 with the parent's tensor keys.
- `collect_p.py`, `dirs_p.py`, `tf_p.py`, `build_p.py`: the direction-ablation study on cd0b95bb that was killed
  (negative result, kept for reproducibility).

Results, evaluation IDs and seeds will be added before any submission.

## Night results (2026-11-07, proxy400 = math500/gsm8k/mmlu_pro/arc_c, 3 samples/prompt, server sampler)
Reference 9492e219: seed set 1+2 acc 45.4, truncation 11.9%, accuracy on finished answers 53.0.

| candidate | result | verdict |
|---|---|---|
| GRPO run1: lr 5e-6, group-std advantage | multi-`</think>` hack from step 5 (39% → 100% of rollouts), thinking median 490 → 36 tokens by step 30 | collapsed |
| GRPO lr 1e-5 | 95% truncated rollouts at step 3 | unstable |
| GRPO variant B: lr 3e-6, mean advantage, DAPO overlong shaping | step 15: acc 43.1, **truncation 33.0%**, acc on finished 63.9 — learns to think longer | worse |
| GRPO + Kimi in-group length reward (w 0.3 / 0.15), hard pool | skipped thinking in 50–70% of rollouts, multi-close returns | collapsed |
| ExPO 9492e219 + a(9492e219 − 344c8376), a 0.5 / 1.0 | acc −5.6 ± 1.7 / −13.3 ± 1.9, truncation −2 | worse |
| **9492e219 + final norm ×1.05** | s1+s2: acc **46.8** (paired **+1.41 ± 1.11**), truncation 12.0%, acc on finished 55.4 | submitted (probe) |
| 9492e219 + final norm ×1.025 | s1: acc −1.56 ± 1.47 | not submitted |

The ×1.05 final-norm gain is poppingbean's edit (ilblackdragon/deltaleague#43, which put it on our 344c8376); on the server
it gave +1.4 on b8af1dbc (27e49fd4) and +2.2 on 344c8376 (cd0b95bb). Here it is applied to ma4ypic4y's GRPO model.
Patch: one tensor (`model.language_model.norm.weight`, (1+w) × 1.05), parent 9492e219-6f07-4bdc-b863-e86274ee62c2.
Seed set 2 of the candidate ran on an RTX 3090 while the reference ran on an RTX 5070 Ti (vLLM numerics can differ).
Lesson: continuing GRPO from 9492e219 with our prompt pools is unstable within 5–15 steps; format guards (single
`</think>`) are needed from step 1 and length shaping in either direction is harmful.

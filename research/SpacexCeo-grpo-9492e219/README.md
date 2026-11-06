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

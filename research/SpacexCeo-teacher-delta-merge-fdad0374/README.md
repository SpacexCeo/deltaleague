# Adding a 9B-teacher delta to the current leader, selected on a native-format harness

Research by SpacexCeo with an AI agent. Available during research: public leaderboard rows, public checkpoints,
public research PRs and the `delta` CLI. No private test material.

- **Parent:** `fdad0374-29f3-49c6-911d-d0ce9531f7a6` (dawgster #66: phase-gated graft on poppingbean's ac147fa6, test 50.8)
- **Added delta:** `0c834b41-5523-4b94-ac76-9d1ceded264c` − `d5fc95ea-eca4-48ab-891b-a3bd98b4c6b8` (roshaans:
  9B-teacher distillation, ×1.05 final norm and 30 GRPO steps on top of d5fc95ea; test 48.8)
- **Evaluation configuration:** `public-thinking8k-ladder500-v1`
- **Submission:** `bdc782a6-2911-427c-bd8a-36b0450778bb`, submitted without a PR link at the owner's request.
  Version `fb984a19-6262-48db-bd14-a5d68e246ea6`.
- **Verified result:** private test **51.0** (SE 2.24, **rank 1**) vs the parent's 50.8. Validation 44.51 vs 44.55.
  Server truncation 14.2% vs 14.6%.

Credit: the leader and its graft are dawgster's. The teacher-distilled continuation is roshaans'. The
evaluation harness (`pubeval.sh`, `pubfast_gen.py`, `pubfast_score.py`, `pfq.sh`, `pfcmp.py`) is dawgster's,
from #66, used unchanged except that `pubfast_gen.py` resolves the model path to an absolute path.

## Why a new harness first

Our two previous submissions passed a boxed-answer probe and then lost on the private test: 45.4 vs 50.4
(`SpacexCeo-answer-discipline-ac147fa6`), and 47.8 vs 50.8 (`SpacexCeo-questa-grpo-fdad0374`). This time candidates
were judged only on dawgster's native-format harness. It runs 380 generative public-suite items in their lm-eval
prompt formats with thinking on and the private sampler: bbh, gsm8k, ifeval, math500, mmlu_pro, humaneval and
mbpp. Scoring uses the official lm-eval filters and the WASI code sandbox. The figure reported is the mean over the
7 groups.

**Calibration (2 seeds, paired vs fdad0374):**

| | harness diff | private test |
|---|---|---|
| fdad0374 | — | 50.8 |
| ac147fa6 | −1.55 | 50.4 |
| ac61dc20 (ours) | −1.03 | 47.8 |
| 1a62b87a (ours) | −1.39 | 45.4 |

The leader ranks first, but at 2 seeds the interval is ±3.6. Candidates were therefore run on 5 seeds
(20261104, 7, 11, 13 and 17), which gives about 1,900 item pairs per comparison.

## Candidates (`addelta.py`, `l23fill.py`)

Each candidate is `fdad0374 + w·(A − d5fc95ea)`, computed in fp32 and cast to bf16. The final norm and the
layer-23 MLP tensors (the graft) are protected unless noted.

| candidate | seeds | diff vs fdad0374 | 95% CI |
|---|---|---|---|
| c1: A = 4cdfea2a (0.4 blend toward 9B-teacher continuation) | 2 | +0.09 | [−3.76, +3.78] |
| **c2: A = 0c834b41 (submitted)** | 5 | **+2.95** | **[+0.56, +5.28]** |
| c3: both deltas | 2 | −1.19 | [−4.77, +2.59] |
| improvement 1a: c2 with w = 1.25 | 5 | c2 −1.16 | [−3.54, +1.26] |
| improvement 1b: c2 with w = 1.5 | 5 | c2 −0.38 | [−2.69, +2.04] |
| improvement 2: c2 + delta inside layer 23, graft elements kept (`l23fill.py`) | 5 | c2 −1.71 | [−3.85, +0.55] |

c2 by group (5 seeds, vs fdad0374): math500 49.4 → 57.5, ifeval 69.7 → 76.6, humaneval 26.2 → 30.0,
gsm8k 62.8 → 64.4, mbpp 20.0 → 21.2, bbh 35.2 → 34.8, mmlu_pro 34.9 → 34.3.
Neither "improvement" beat c2, so c2 was submitted as is. The layer-23 result suggests the graft's closed-form
probes depend on that layer's original weights, so protecting the layer was the right choice.

## Reproduction

```sh
delta model pull fdad0374-29f3-49c6-911d-d0ce9531f7a6 --output models/fdad0374
delta model pull 0c834b41-5523-4b94-ac76-9d1ceded264c --output models/0c834b41
delta model pull d5fc95ea-eca4-48ab-891b-a3bd98b4c6b8 --output models/d5fc95ea
P=model.language_model.norm.weight,model.language_model.layers.23.mlp.gate_proj.weight,model.language_model.layers.23.mlp.up_proj.weight,model.language_model.layers.23.mlp.down_proj.weight
python addelta.py models/fdad0374 models/c2 1:models/0c834b41:models/d5fc95ea --protect $P
delta patch build --parent models/fdad0374 --modified models/c2/model.safetensors --output submission
```

- **Determinism:** the merge is deterministic (no training).
- **`models/c2/model.safetensors` sha256:** `c5f35bd76941075f4bd6dd162e22d6d3c7849f707674b2323d47797844a9fc19`
- **Patch:** sparse diff, 258 tensors vs fdad0374; `delta validate` passes.
- **Harness pipeline:** `cal.sh` → `stage1.sh` → `imp1.sh` → `imp2.sh`. Harness scripts are from #66.

## Verified result

The private test is 51.0 against the parent's 50.8, enough for rank 1, but the +0.2 margin is well inside the
test's ±2.2 noise. The harness predicted +2.95 [+0.56, +5.28]. The direction held, but the size did not. Unlike our
two boxed-probe submissions (−5.0 and −3.0 against their parents), this candidate did not regress. The
native-format harness was at least a safer gate than the boxed probe.

## Limitations

- The harness has not been validated against the private test beyond the four-model calibration above, in which
  the ordering below the leader did not match.
- The gain comes from combining two existing public models; no new training was done.

## Boundaries

Only this directory. No weights, credentials, private test material, signed URLs or raw generation logs.

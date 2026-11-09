"""Rescore probe rows on the post-</think> answer only (from the saved tail), then pair. usage: rescore.py BASE CAND [BASE CAND ...]"""
import json, sys, statistics, math
from probe import items
from delta_cli.scoring import score_answer
cache = {}
def fix(f, seed):
    if seed not in cache: cache[seed] = items(150, seed)
    its = cache[seed]; out = []
    for l in open(f):
        r = json.loads(l)
        t = r.get("tail") or r.get("text") or ""
        if r["truncated"] or r["close"] is None: r["score2"] = 0.0
        elif "</think>" not in t: r["score2"] = None
        else:
            try: r["score2"] = score_answer(its[r["k"]], t.split("</think>")[-1].split("<|im_end|>")[0])
            except Exception: r["score2"] = 0.0
        out.append(r)
    return out
args = sys.argv[1:]; allD = []
for b, c in zip(args[0::2], args[1::2]):
    seed = int(b.split("_s")[-1].split(".")[0])
    A, B = fix(b, seed), fix(c, seed)
    pairs = [(x, y) for x, y in zip(A, B) if x["score2"] is not None and y["score2"] is not None]
    print(f"  usable pairs {len(pairs)} of {len(A)}")
    A = [x for x, _ in pairs]; B = [y for _, y in pairs]
    d = [y["score2"] - x["score2"] for x, y in zip(A, B)]; allD += d
    print(f"{c}: base={100*statistics.mean(x['score2'] for x in A):.1f} cand={100*statistics.mean(y['score2'] for y in B):.1f} diff={100*statistics.mean(d):+.2f}±{100*statistics.stdev(d)/math.sqrt(len(d)):.2f} (old-scoring base={100*statistics.mean(x['score'] for x in A):.1f})")
print(f"POOLED n={len(allD)} diff={100*statistics.mean(allD):+.2f}±{100*statistics.stdev(allD)/math.sqrt(len(allD)):.2f}")

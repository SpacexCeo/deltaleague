import json, sys, statistics, math
def load(f): return {(r["k"], i): r for i, r in enumerate(json.loads(l) for l in open(f))}
def post(r): return r["truncated"] and r["close"] is not None
for base, cand in zip(sys.argv[1::2][:len(sys.argv)//2], sys.argv[2::2]):
    A, B = load(base), load(cand)
    keys = sorted(set(A) & set(B))
    for src in sorted({A[k]["src"] for k in keys}) + [None]:
        ks = [k for k in keys if src is None or A[k]["src"] == src]
        d = [B[k]["score"] - A[k]["score"] for k in ks]
        se = statistics.stdev(d) / math.sqrt(len(d))
        print(f"{base}->{cand} {src or 'ALL':8} n={len(ks)} base={100*statistics.mean(A[k]['score'] for k in ks):.1f} "
              f"cand={100*statistics.mean(B[k]['score'] for k in ks):.1f} diff={100*statistics.mean(d):+.2f}±{100*se:.2f} "
              f"trunc {100*statistics.mean(A[k]['truncated'] for k in ks):.1f}->{100*statistics.mean(B[k]['truncated'] for k in ks):.1f} "
              f"post-think-trunc {sum(post(A[k]) for k in ks)}->{sum(post(B[k]) for k in ks)} "
              f"medtok {statistics.median(A[k]['tokens'] for k in ks):.0f}->{statistics.median(B[k]['tokens'] for k in ks):.0f}")

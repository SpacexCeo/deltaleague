"""Thinking-mode probe at the private-test sampler, with a simulated 'deadline exit'.
Stage A: normal generation (max 8192). Stage B (simulation): for every sample whose </think> is not closed
within D tokens, cut at D, append EXIT text, and let the model answer within the remaining budget.
usage: probe.py MODEL OUT.jsonl [--deadline 6144] [--n-per 150] [--samples 1] [--seed 0]"""
import argparse, json, os, random, statistics, time
from datasets import load_dataset
from vllm import LLM, SamplingParams
from delta_cli.scoring import score_answer

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
CSUFFIX = "\nPlease reason step by step, and put the letter of the correct option within \\boxed{}."
L = "ABCDE"
EXITS = {
 "commit": "\n\nOkay, I have spent enough time on this. I'll commit to my best answer now.\n</think>\n\n",
 "bare": "\n</think>\n\n",
 "none": "",
}

def boxed(s):
    i = s.rfind("\\boxed{")
    if i < 0: return s
    d, j0 = 0, i + 7
    for j in range(j0, len(s)):
        d += {"{": 1, "}": -1}.get(s[j], 0)
        if d < 0: return s[j0:j]
    return s[j0:]

def mc(q, opts, ans, src):
    return {"prompt": q + "\n\n" + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(opts)) + CSUFFIX,
            "scorer": "choice", "answer": L[ans], "src": src}

def items(n, seed):
    rng = random.Random(seed); out = []
    g = load_dataset("openai/gsm8k", "main", split="test")
    out += [{"prompt": g[i]["question"] + SUFFIX, "scorer": "math", "answer": g[i]["answer"].split("####")[-1].strip(), "src": "gsm8k"}
            for i in rng.sample(range(len(g)), n)]
    m500 = {d["problem"].strip() for d in load_dataset("HuggingFaceH4/MATH-500", split="test")}
    m = [d for d in load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
         if d["problem"].strip() not in m500 and d["level"] in ("Level 3", "Level 4", "Level 5")]
    out += [{"prompt": d["problem"] + SUFFIX, "scorer": "math", "answer": boxed(d["solution"]), "src": "math"} for d in rng.sample(m, n)]
    a = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for d in (a[i] for i in rng.sample(range(len(a)), n)):
        lab = d["choices"]["label"]
        if d["answerKey"] in lab and len(lab) <= 5:
            out.append(mc(d["question"], d["choices"]["text"], lab.index(d["answerKey"]), "arc_c"))
    u = load_dataset("cais/mmlu", "all", split="test")
    for d in (u[i] for i in rng.sample(range(len(u)), n)):
        out.append(mc(d["question"], d["choices"], d["answer"], "mmlu"))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("out")
    p.add_argument("--deadline", type=int, nargs="*", default=[])
    p.add_argument("--exit", default="commit")
    p.add_argument("--n-per", type=int, default=150)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--temp", type=float, default=1.0); p.add_argument("--pp", type=float, default=1.5)
    a = p.parse_args()
    its = items(a.n_per, a.seed)
    llm = LLM(os.path.abspath(a.model), language_model_only=True, dtype="bfloat16", max_model_len=a.max_tokens + 2048, seed=0,
              gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.85")), max_num_seqs=256)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
               add_generation_prompt=True, enable_thinking=True) for it in its]
    sp = SamplingParams(n=a.samples, temperature=a.temp, top_p=0.95, top_k=20, presence_penalty=a.pp,
                        max_tokens=a.max_tokens, seed=1000 + a.seed, skip_special_tokens=False)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"stage A {time.time()-t0:.0f}s", flush=True)
    end_id = tok.convert_tokens_to_ids("</think>")
    rows = []
    for k, (it, pr, o) in enumerate(zip(its, prompts, outs)):
        for c in o.outputs:
            ids = list(c.token_ids)
            close = ids.index(end_id) if end_id in ids else None
            tr = len(ids) >= a.max_tokens
            sc = 0.0 if (tr or close is None) else score_answer(it, c.text)
            rows.append({"k": k, "src": it["src"], "tokens": len(ids), "close": close, "truncated": tr, "score": sc,
                         "ids": ids, "prompt": pr, **({"text": c.text} if tr else {"tail": c.text[-6000:]})})
    exit_text = EXITS[a.exit]; exit_ids = tok(exit_text, add_special_tokens=False)["input_ids"]
    for D in a.deadline:
        need = [r for r in rows if r["close"] is None or r["close"] >= D]
        reqs = [r["prompt"] + tok.decode(r["ids"][:D], skip_special_tokens=False) + exit_text for r in need]
        budget = a.max_tokens - D - len(exit_ids)
        sp2 = SamplingParams(n=1, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=budget,
                             seed=2000 + a.seed, skip_special_tokens=False)
        o2 = llm.generate(reqs, sp2) if reqs else []
        for r, g in zip(need, o2):
            t = g.outputs[0]
            ok = len(t.token_ids) < budget
            r[f"d{D}"] = score_answer(its[r["k"]], "</think>" + t.text) if ok else 0.0
            r[f"d{D}_tok"] = D + len(exit_ids) + len(t.token_ids)
        for r in rows:
            if f"d{D}" not in r: r[f"d{D}"] = r["score"]
    with open(a.out, "w") as f:
        for r in rows:
            r = {k: v for k, v in r.items() if k not in ("ids", "prompt")}
            f.write(json.dumps(r) + "\n")
    for src in sorted({r["src"] for r in rows}) + [None]:
        s = [r for r in rows if src is None or r["src"] == src]
        line = (f"{src or 'ALL':6} n={len(s)} acc={100*statistics.mean(r['score'] for r in s):.1f} "
                f"trunc={100*statistics.mean(r['truncated'] for r in s):.1f} med_tok={statistics.median(r['tokens'] for r in s):.0f}")
        for D in a.deadline:
            late = [r for r in s if r["close"] is None or r["close"] >= D]
            line += f" | D{D}: acc={100*statistics.mean(r[f'd{D}'] for r in s):.1f} (late={len(late)} rescued={sum(r[f'd{D}'] for r in late):.0f} lost={sum(r['score'] for r in late):.0f})"
        print(line, flush=True)
    print(f"total {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()

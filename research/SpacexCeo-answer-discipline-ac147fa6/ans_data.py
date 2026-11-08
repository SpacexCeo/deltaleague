"""Answer-discipline data: own thinking (closed) + shortest correct concise answer resampled after </think>.
usage: ans_data.py MODEL OUT.jsonl"""
import argparse, json, os, random, time
from datasets import load_dataset
from vllm import LLM, SamplingParams
from delta_cli.scoring import score_answer
from probe import SUFFIX, CSUFFIX, boxed, mc

def pool(seed, n_math, n_gsm, n_mc):
    rng = random.Random(seed); out = []
    g = load_dataset("openai/gsm8k", "main", split="train")
    out += [{"prompt": g[i]["question"] + SUFFIX, "scorer": "math", "answer": g[i]["answer"].split("####")[-1].strip(), "src": "gsm8k"}
            for i in rng.sample(range(len(g)), n_gsm)]
    m = [d for d in load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train") if d["level"] in ("Level 2", "Level 3", "Level 4", "Level 5")]
    out += [{"prompt": d["problem"] + SUFFIX, "scorer": "math", "answer": boxed(d["solution"]), "src": "math"} for d in rng.sample(m, n_math)]
    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    for i in rng.sample(range(len(aux)), n_mc):
        d = aux[i]; d = d.get("train", d)
        out.append(mc(d["question"], d["choices"], d["answer"], "mmlu_aux"))
    rng.shuffle(out); return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("model"); p.add_argument("out")
    p.add_argument("--n-math", type=int, default=1100); p.add_argument("--n-gsm", type=int, default=700); p.add_argument("--n-mc", type=int, default=300)
    p.add_argument("--k", type=int, default=6); p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    its = pool(a.seed, a.n_math, a.n_gsm, a.n_mc)
    llm = LLM(os.path.abspath(a.model), language_model_only=True, dtype="bfloat16", max_model_len=8192 + 2048, seed=0,
              gpu_memory_utilization=0.85, max_num_seqs=256)
    tok = llm.get_tokenizer(); end_id = tok.convert_tokens_to_ids("</think>")
    prompts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
               add_generation_prompt=True, enable_thinking=True) for it in its]
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=8192, seed=3000 + a.seed, skip_special_tokens=False)
    t0 = time.time(); outs = llm.generate(prompts, sp); print(f"rollouts {time.time()-t0:.0f}s", flush=True)
    cands = []; post_trunc = 0
    for it, pr, o in zip(its, prompts, outs):
        ids = list(o.outputs[0].token_ids)
        if end_id not in ids: continue
        c = ids.index(end_id)
        if c > 6500: continue
        tr = len(ids) >= 8192; post_trunc += tr
        ans_len = len(ids) - c - 1
        orig = 0.0 if tr else score_answer(it, o.outputs[0].text)
        cands.append({"it": it, "prompt": pr, "prefix": tok.decode(ids[:c + 1], skip_special_tokens=False), "close": c,
                      "orig_ans_len": ans_len, "orig_trunc": tr, "orig_score": orig})
    print(f"closed {len(cands)} post-think truncated {post_trunc}", flush=True)
    sp2 = SamplingParams(n=a.k, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=1024, seed=4000 + a.seed, skip_special_tokens=False)
    o2 = llm.generate([c["prompt"] + c["prefix"] for c in cands], sp2)
    kept = 0
    with open(a.out, "w") as f:
        for c, g in zip(cands, o2):
            good = []
            for t in g.outputs:
                if t.finish_reason != "stop": continue
                txt = t.text.split("<|im_end|>")[0]
                if "<think>" in txt or "</think>" in txt: continue
                if score_answer(c["it"], "</think>" + txt): good.append(txt)
            c["n_correct"] = len(good); c["answer"] = min(good, key=len) if good else None
            kept += bool(good)
            c.pop("it")
            f.write(json.dumps(c) + "\n")
    print(f"kept {kept} of {len(cands)} total {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()

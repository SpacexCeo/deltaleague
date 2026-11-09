"""IFEval (google/IFEval, 541 prompts) in thinking mode at the private-test sampler; strict prompt-level accuracy
on the text after </think> (unclosed or truncated = 0). usage: ifprobe.py MODEL OUT.jsonl [--seed 0]"""
import argparse, json, os, statistics
from datasets import load_dataset
from vllm import LLM, SamplingParams
from lm_eval.tasks.ifeval.utils import process_results
def main():
    p = argparse.ArgumentParser(); p.add_argument("model"); p.add_argument("out"); p.add_argument("--seed", type=int, default=0); a = p.parse_args()
    docs = list(load_dataset("google/IFEval", split="train"))
    llm = LLM(os.path.abspath(a.model), language_model_only=True, dtype="bfloat16", max_model_len=10240, seed=0, gpu_memory_utilization=0.85, max_num_seqs=256)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for d in docs]
    sp = SamplingParams(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=8192, seed=1000 + a.seed, skip_special_tokens=False)
    outs = llm.generate(prompts, sp)
    rows = []
    for d, o in zip(docs, outs):
        c = o.outputs[0]; t = c.text
        ok = "</think>" in t and len(c.token_ids) < 8192
        ans = t.split("</think>")[-1].split("<|im_end|>")[0].strip() if ok else ""
        sc = float(process_results(d, [ans])["prompt_level_strict_acc"]) if ok else 0.0
        rows.append({"k": d["key"], "src": "ifeval", "score": sc, "truncated": len(c.token_ids) >= 8192, "close": t.find("</think>") if "</think>" in t else None, "tokens": len(c.token_ids)})
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(f"IFEVAL n={len(rows)} strict={100*statistics.mean(r['score'] for r in rows):.1f} trunc={100*statistics.mean(r['truncated'] for r in rows):.1f}")

if __name__ == "__main__":
    main()

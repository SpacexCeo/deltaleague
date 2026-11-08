"""LoRA 'deadline reflex' training. Loss = presence-penalty-aware CE on the exit (+answer) tokens after a long prefix
+ lam * KL(base || tuned) at earlier anchor positions (paragraph breaks before ANCHOR_MAX), so the model learns to exit
only late. Merges LoRA and writes a full checkpoint with the parent's tensor names/dtypes.
usage: train_exit.py PARENT_DIR DATA.jsonl OUT_DIR [--epochs 2] [--lr 1e-4] [--rank 16] [--lam 1.0]"""
import argparse, json, random, shutil, math
from pathlib import Path
import torch, torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model
from probe import EXITS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("parent"); p.add_argument("data"); p.add_argument("out")
    p.add_argument("--epochs", type=float, default=2); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16); p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--accum", type=int, default=8); p.add_argument("--exit", default="commit")
    p.add_argument("--anchor-max", type=int, default=4600); p.add_argument("--n-anchor", type=int, default=192)
    p.add_argument("--exit-only-frac", type=float, default=0.5, help="share of no-correct-answer items kept with loss on EXIT only")
    p.add_argument("--presence", type=float, default=1.5); p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    par = Path(a.parent); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(par)
    cfg = AutoConfig.from_pretrained(par)
    Auto = AutoModelForImageTextToText if type(cfg) in AutoModelForImageTextToText._model_mapping else AutoModelForCausalLM
    model = Auto.from_pretrained(par, dtype=torch.bfloat16).cuda()
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model.layers" in n]
    targets = sorted({n.split(".")[-1] for n in names})
    print("lora targets", targets, len(names), flush=True)
    lcfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                      target_modules=[n for n in names])
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    exit_text = EXITS[a.exit]
    rows = [json.loads(l) for l in open(a.data)]
    data = []
    for r in rows:
        if r["answer"] is None and (a.exit_only_frac <= 0 or rng.random() > a.exit_only_frac): continue
        p_ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        pre = tok(r["prefix"], add_special_tokens=False)["input_ids"]
        ex = tok(exit_text, add_special_tokens=False)["input_ids"]
        ans = tok(r["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"] if r["answer"] is not None else []
        ids = p_ids + pre + ex + ans
        n_p = len(p_ids)
        tgt = list(range(n_p + len(pre), len(ids)))  # positions predicted (token index)
        # anchors: paragraph-break positions early in the completion + random early positions
        early_end = n_p + min(len(pre), a.anchor_max)
        brk = [n_p + j + 1 for j, t in enumerate(pre[:a.anchor_max]) if "\n\n" in tok.decode([t])]
        rnd = [rng.randrange(n_p + 1, early_end) for _ in range(a.n_anchor)]
        anc = sorted(set(brk + rnd))
        data.append((ids, n_p, tgt, anc))
    print(f"train items {len(data)} (with answer {sum(1 for r in rows if r['answer'])})", flush=True)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    total = math.ceil(len(data) * a.epochs / a.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / max(total, 1)))
    lm = model.base_model.model
    embed_w = lm.get_input_embeddings().weight
    def hidden_states(ids):
        inner = lm.model.language_model if hasattr(lm.model, "language_model") else lm.model
        return inner(input_ids=ids).last_hidden_state[0]
    def pen_logits(h, ids_list, n_p, pos):
        # logits at positions pos (predicting token pos), with presence penalty for tokens seen in completion before pos
        logits = (h[torch.tensor(pos, device="cuda") - 1] @ embed_w.T).float()
        comp = torch.tensor(ids_list[n_p:], device="cuda")
        first = torch.full((embed_w.shape[0],), 10**9, dtype=torch.long, device="cuda")
        first.scatter_reduce_(0, comp, torch.arange(len(comp), device="cuda") + n_p, reduce="amin")
        seen = first[None, :] < torch.tensor(pos, device="cuda")[:, None]
        return logits - a.presence * seen.float()
    step = 0; order = []
    while len(order) < len(data) * a.epochs:
        o = list(range(len(data))); rng.shuffle(o); order += o
    order = order[:int(len(data) * a.epochs)]
    model.train(); logs = []
    for i, k in enumerate(order):
        ids, n_p, tgt, anc = data[k]
        x = torch.tensor([ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = hidden_states(x)
        lt = pen_logits(h, ids, n_p, tgt)
        ce = F.cross_entropy(lt, x[0, tgt])
        la = pen_logits(h, ids, n_p, anc)
        with torch.no_grad(), model.disable_adapter():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hb = hidden_states(x)
            lb = pen_logits(hb, ids, n_p, anc)
        kl = F.kl_div(la.log_softmax(-1), lb.log_softmax(-1), log_target=True, reduction="batchmean")
        loss = (ce + a.lam * kl) / a.accum
        loss.backward()
        logs.append((ce.item(), kl.item()))
        if (i + 1) % a.accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            c = sum(x[0] for x in logs) / len(logs); kk = sum(x[1] for x in logs) / len(logs); logs = []
            print(json.dumps({"step": step, "of": total, "ce": round(c, 4), "kl": round(kk, 5), "lr": sched.get_last_lr()[0]}), flush=True)
    model = model.merge_and_unload()
    sd = model.state_dict()
    with safe_open(str(par / "model.safetensors"), "pt") as f:
        keys = list(f.keys())
        state = {}
        changed = 0
        for kk in keys:
            orig = f.get_tensor(kk)
            if kk in sd:
                t = sd[kk].detach().to(orig.dtype).cpu().contiguous()
                changed += int(not torch.equal(t, orig))
                state[kk] = t
            else:
                state[kk] = orig
    print(f"tensors changed {changed} of {len(keys)}", flush=True)
    for fl in par.iterdir():
        if fl.is_file() and fl.name not in ("model.safetensors", "delta-model.json"):
            shutil.copy(fl, out / fl.name)
    save_file(state, str(out / "model.safetensors"), metadata={"format": "pt"})
    print("saved", out, flush=True)

if __name__ == "__main__":
    main()

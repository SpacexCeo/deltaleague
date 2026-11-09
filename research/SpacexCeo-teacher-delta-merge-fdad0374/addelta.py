"""out = base + sum_i w_i * (A_i - B_i), in float32, cast to base dtype. Protected tensors keep the base value.
usage: addelta.py BASE_DIR OUT_DIR w:A_DIR:B_DIR [w:A_DIR:B_DIR ...] [--protect name1,name2]"""
import sys, shutil, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
args = sys.argv[1:]; protect = set()
if "--protect" in args:
    i = args.index("--protect"); protect = set(args[i + 1].split(",")); args = args[:i] + args[i + 2:]
base, out, specs = Path(args[0]), Path(args[1]), args[2:]
out.mkdir(parents=True, exist_ok=True)
for f in base.iterdir():
    if f.is_file() and f.name not in ("model.safetensors", "delta-model.json"): shutil.copy(f, out / f.name)
hs = [(float(w), safe_open(str(Path(a) / "model.safetensors"), "pt"), safe_open(str(Path(b) / "model.safetensors"), "pt"))
      for w, a, b in (s.split(":") for s in specs)]
res, changed = {}, 0
with safe_open(str(base / "model.safetensors"), "pt") as fb:
    for k in fb.keys():
        t = fb.get_tensor(k)
        if k in protect or not t.is_floating_point():
            res[k] = t; continue
        acc = t.float()
        for w, ha, hb in hs:
            acc = acc + w * (ha.get_tensor(k).float() - hb.get_tensor(k).float())
        n = acc.to(t.dtype)
        changed += int(not torch.equal(n, t)); res[k] = n
save_file(res, str(out / "model.safetensors"), metadata={"format": "pt"})
print(out, "tensors changed", changed, "of", len(res))

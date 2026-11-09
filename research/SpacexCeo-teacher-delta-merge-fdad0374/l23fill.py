"""Apply w*(A-B) to the layer-23 MLP tensors of MODEL only where GRAFTED == UNGRAFTED (leaves the graft neurons intact).
usage: l23fill.py MODEL_DIR OUT_DIR w A_DIR B_DIR GRAFTED_DIR UNGRAFTED_DIR"""
import sys, shutil, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
m, out, w, A, B, G, U = sys.argv[1:8]; w = float(w); m, out = Path(m), Path(out)
out.mkdir(parents=True, exist_ok=True)
for f in m.iterdir():
    if f.is_file() and f.name not in ("model.safetensors", "delta-model.json"): shutil.copy(f, out / f.name)
names = [f"model.language_model.layers.23.mlp.{n}.weight" for n in ("gate_proj", "up_proj", "down_proj")]
op = lambda d: safe_open(str(Path(d) / "model.safetensors"), "pt")
hm, ha, hb, hg, hu = op(m), op(A), op(B), op(G), op(U)
res = {}
for k in hm.keys():
    t = hm.get_tensor(k)
    if k in names:
        same = hg.get_tensor(k) == hu.get_tensor(k)
        new = (t.float() + w * (ha.get_tensor(k).float() - hb.get_tensor(k).float()) * same.float()).to(t.dtype)
        print(k, "graft elements kept:", int((~same).sum()), "elements changed:", int((new != t).sum()))
        t = new
    res[k] = t
save_file(res, str(out / "model.safetensors"), metadata={"format": "pt"})
print("saved", out)

#!/bin/bash
cd /root/h
P=model.language_model.norm.weight,model.language_model.layers.23.mlp.gate_proj.weight,model.language_model.layers.23.mlp.up_proj.weight,model.language_model.layers.23.mlp.down_proj.weight
.venv/bin/python scripts/addelta.py models/fdad0374 models/c2w125 1.25:models/0c834b41:models/d5fc95ea --protect $P
.venv/bin/python scripts/addelta.py models/fdad0374 models/c2w15 1.5:models/0c834b41:models/d5fc95ea --protect $P
scripts/pfq.sh "c2w15:20261104 c2w125:20261104 c2w15:7 c2w125:7 c2w15:11 c2w125:11 c2w15:13 c2w125:13 c2w15:17 c2w125:17"
for m in c2w125 c2w15; do python3 scripts/pfcmp.py c2 $m 20261104,7,11,13,17; done
echo IMP1_DONE

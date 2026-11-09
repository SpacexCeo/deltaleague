#!/bin/bash
cd /root/h
until grep -q CAL_DONE /root/cal.log; do sleep 20; done
for v in 4cdfea2a-851b-4662-be8c-6af99bb2194c d5fc95ea-eca4-48ab-891b-a3bd98b4c6b8 0c834b41-5523-4b94-ac76-9d1ceded264c; do s=${v:0:8}; [ -f models/$s/model.safetensors ] || /root/v/bin/delta model pull $v --output models/$s > pull_$s.log 2>&1; done
P=model.language_model.norm.weight,model.language_model.layers.23.mlp.gate_proj.weight,model.language_model.layers.23.mlp.up_proj.weight,model.language_model.layers.23.mlp.down_proj.weight
cp /root/addelta.py scripts/
.venv/bin/python scripts/addelta.py models/fdad0374 models/c1 1:models/4cdfea2a:models/d5fc95ea --protect $P
.venv/bin/python scripts/addelta.py models/fdad0374 models/c2 1:models/0c834b41:models/d5fc95ea --protect $P
.venv/bin/python scripts/addelta.py models/fdad0374 models/c3 1:models/4cdfea2a:models/d5fc95ea 1:models/0c834b41:models/d5fc95ea --protect $P
scripts/pfq.sh "c1:20261104 c2:20261104 c3:20261104 c1:7 c2:7 c3:7"
for m in c1 c2 c3; do python3 scripts/pfcmp.py fdad0374 $m; done
echo STAGE1_DONE

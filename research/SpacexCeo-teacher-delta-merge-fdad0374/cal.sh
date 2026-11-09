#!/bin/bash
# calibration: official run on leader (for prompts), then pubfast for 4 known models x 2 seeds
set -x
cd /root/h
ln -sfn /root/v .venv; ln -sfn /root/vpe .venv-pe; mkdir -p runs models scripts pe
cp /root/*.py /root/*.sh scripts/ 2>/dev/null
chmod +x scripts/*.sh
[ -d ws ] || /root/v/bin/delta competition init qwen35-08b-v1 --output ws > init.log 2>&1
cp ws/validation.json pe/validation.json; cp ws/evaluation.json pe/evaluation.json
ln -sfn /root/pe/wasi pe/wasi
while read short full; do [ -f models/$short/model.safetensors ] || /root/v/bin/delta model pull $full --output models/$short > pull_$short.log 2>&1; done < /root/ids.txt
ln -sfn models/fdad0374 models/leader
[ -f runs/pub_leader/result.json ] || scripts/pubeval.sh models/fdad0374 leader > pubeval_leader.log 2>&1
ls runs/pub_leader/reports
scripts/pfq.sh "fdad0374:20261104 ac147fa6:20261104 ac61dc20:20261104 1a62b87a:20261104 fdad0374:7 ac147fa6:7 ac61dc20:7 1a62b87a:7"
for m in ac147fa6 ac61dc20 1a62b87a; do python3 scripts/pfcmp.py fdad0374 $m; done
echo CAL_DONE

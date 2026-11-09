#!/bin/bash
cd /root/h
until grep -q IMP1_DONE /root/imp1.log; do sleep 20; done
cp /root/l23fill.py scripts/
# pick strength: highest 5-seed mean among c2 (1.0), c2w125, c2w15
best=$(python3 - <<PY
import re
txt=open('/root/imp1.log').read()
cands={'c2':(1.0,0.0)}
for m,w in (('c2w125',1.25),('c2w15',1.5)):
    r=re.search(r'^c2 [\d.]+ -> %s [\d.]+  diff ([+-][\d.]+)'%m, txt, re.M)
    if r: cands[m]=(w,float(r.group(1)))
m=max(cands,key=lambda k:cands[k][1]); print(m, cands[m][0])
PY
)
set -- $best; BM=$1; BW=$2; echo "IMP1 winner: $BM w=$BW"
.venv/bin/python scripts/l23fill.py models/$BM models/c4 $BW models/0c834b41 models/d5fc95ea models/fdad0374 models/ac147fa6
scripts/pfq.sh "c4:20261104 c4:7 c4:11 c4:13 c4:17"
python3 scripts/pfcmp.py $BM c4 20261104,7,11,13,17
python3 scripts/pfcmp.py fdad0374 c4 20261104,7,11,13,17
python3 scripts/pfcmp.py fdad0374 $BM 20261104,7,11,13,17
echo IMP2_DONE

"""Pick training traces: per prompt, among samples that are correct, closed after >= 50 thinking tokens and not truncated,
keep the shortest (and optionally a second one). usage: select_rft.py OUT.jsonl IN1.jsonl [IN2 ...] [--per 1] [--max_tokens 6000]"""
import json, sys, collections
args = [a for a in sys.argv[1:] if not a.startswith('--')]
per = int(sys.argv[sys.argv.index('--per') + 1]) if '--per' in sys.argv else 1
mx = int(sys.argv[sys.argv.index('--max_tokens') + 1]) if '--max_tokens' in sys.argv else 6000
args = [a for a in args if not a.isdigit()]
out, ins = args[0], args[1:]
by = collections.defaultdict(list); stats = collections.Counter()
for f in ins:
    for l in open(f):
        r = json.loads(l); stats['gens'] += 1; stats[r['src']] += 1
        if r['score'] == 1 and r['closed'] and not r['truncated'] and r['think_tokens'] >= 50 and r['tokens'] <= mx:
            by[r['id']].append(r)
keep = []
for i, rs in by.items():
    rs.sort(key=lambda r: r['tokens']); keep += rs[:per]
src = collections.Counter(r['src'] for r in keep)
with open(out, 'w') as f:
    for r in keep: f.write(json.dumps({'id': r['id'], 'src': r['src'], 'prompt': r['rendered_prompt'], 'completion': r['text']}) + '\n')
print('gens', stats['gens'], 'prompts with a keeper', len(by), 'kept', len(keep), dict(src),
      'mean tokens', round(sum(r['tokens'] for r in keep) / max(1, len(keep))))

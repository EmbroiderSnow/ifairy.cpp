"""Summarize saved probe traces and bench JSON; keep large binary traces outside git."""
import hashlib
import json
from pathlib import Path
import struct
import sys

base = Path(__file__).resolve().parent
platform = sys.argv[1]
tmp = Path(sys.argv[2])
if platform == 'phone':
    for path in tmp.iterdir():
        if path.is_file() and path.suffix != '.f32':
            (base/path.name).write_bytes(path.read_bytes())
runs = {}
for path in sorted(base.glob(platform+'-probe-*.stdout.log')):
    name = path.name.removeprefix(platform+'-probe-').removesuffix('.stdout.log')
    record = json.loads(path.read_text())
    runs[name] = {'rss_mib':record['max_rss_native']/1024, 'prompts':[]}
    for i,result in enumerate(record['results']):
        raw = (tmp/f'{platform}-probe-{name}-prompt-{i}.f32').read_bytes()
        values = struct.unpack('<%df' % (len(raw)//4), raw)
        import math
        assert all(map(math.isfinite, values)) and len(values) == result['finite_logits']
        runs[name]['prompts'].append({'sha256':hashlib.sha256(raw).hexdigest(),
                                     'generated_ids':result['generated_ids'], 'finite_logits':len(values)})
comparisons = {}
for a,b in [('auto','repeat'),('auto','lut16'),('auto','serial'),('auto','same-quant-direct'),('auto','off'),('auto','lut-c')]:
    rows = []
    for i in range(2):
        aa=(tmp/f'{platform}-probe-{a}-prompt-{i}.f32').read_bytes()
        bb=(tmp/f'{platform}-probe-{b}-prompt-{i}.f32').read_bytes()
        av=struct.unpack('<32000f',aa[:128000]); bv=struct.unpack('<32000f',bb[:128000])
        d=[abs(x-y) for x,y in zip(av,bv)]
        ga=runs[a]['prompts'][i]['generated_ids']; gb=runs[b]['prompts'][i]['generated_ids']
        rows.append({'trace_equal':aa==bb, 'first_logits_max_abs':max(d), 'first_logits_mean_abs':sum(d)/len(d),
                     'generated_equal':ga==gb, 'first_different_generated_index':next((j for j,(x,y) in enumerate(zip(ga,gb)) if x!=y),None)})
    comparisons[a+'_vs_'+b] = rows
out = {'runs':runs, 'comparisons':comparisons}
(base/(platform+'-correctness-summary.json')).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
print({k:v['rss_mib'] for k,v in runs.items()})

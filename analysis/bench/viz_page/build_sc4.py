# Builds index.html for the sc4 scaling sweep: template.html's <head>/<style> + body_sc4.html, with viz_sc4.json (sweep) and
# viz_data.json (GPU scaling probe, CLIP reference) and Chart.js embedded (the viewer does not reliably serve side files or CDNs).
import json, math, pathlib
here = pathlib.Path(__file__).parent
head = (here / 'template.html').read_text().split('</style>')[0] + '</style>\n'
ref = json.load(open(here / 'viz_data.json'))
data = {'sc4': json.load(open(here / 'viz_sc4.json')), 'ref': {'v3_gpu_scaling': ref.get('v3_gpu_scaling', []), 'clip': ref['notes']['clip_J_per_decision']}}
def clean(o):                                   # NaN/inf (e.g. skipped O-readout counts) are not valid JSON: write null
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    return o
data = json.dumps(clean(data), separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
chart = (here / 'chart.umd.min.js').read_text().replace('</', '<\\/')
html = head + (here / 'body_sc4.html').read_text().replace('/*DATA*/', data).replace('/*CHARTJS*/', chart)
(here / 'index.html').write_text(html)
print('index.html', (here / 'index.html').stat().st_size, 'bytes')

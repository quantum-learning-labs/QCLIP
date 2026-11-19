#!/bin/bash
# Pull the log-depth / IQM outputs from the cluster into adv_data (new files only), re-aggregate, recompute the figure model,
# export the page's store documents (efig_docs/*.json) and render the paper figure on tails. Push the docs with ArtifactData after.
set -uo pipefail
cd "$(dirname "$0")"; source ~/comp_bridge_agent/bin/_hpc.sh
ssh tails-agent "cat /cs/research/pplv/comp_bridge/qclip_wt/energyfig/results/shotcost/ionq_calib/ionq_calib_fit.csv" > /tmp/efi.$$ && [ -s /tmp/efi.$$ ] && cp /tmp/efi.$$ noise_calib_ionq.csv; rm -f /tmp/efi.$$
ssh tails-agent "cat /cs/research/pplv/comp_bridge/qclip_wt/energyfig/results/shotcost/e6_sc4_iqm/iqm_calib_fit.csv" > /tmp/efq.$$ && [ -s /tmp/efq.$$ ] && cp /tmp/efq.$$ noise_calib_iqm.csv; rm -f /tmp/efq.$$
R=qclip_hpc/code/136a567/results/shotcost/adv
for L in 5 6 7; do
  hpc_ssh "cat ~/$R/diag_ef_l$L/structure.csv 2>/dev/null" > /tmp/efs.$$ && [ -s /tmp/efs.$$ ] && cp /tmp/efs.$$ structure_grid_l$L.csv
  hpc_ssh "cat ~/$R/qpu_esp_ef_l$L.csv 2>/dev/null" > /tmp/efs.$$ && [ -s /tmp/efs.$$ ] && { cp /tmp/efs.$$ qpu_esp3_l$L.csv; cp /tmp/efs.$$ qpu_esp4_l$L.csv; }
  hpc_ssh "cat ~/$R/tw_bounds_ef_l$L.csv 2>/dev/null" > /tmp/efs.$$ && [ -s /tmp/efs.$$ ] && cp /tmp/efs.$$ tw_bounds_grid_l$L.csv
  hpc_ssh "cat ~/$R/tw_bounds_ef_ctg_l$L.csv 2>/dev/null" > /tmp/efs.$$ && [ -s /tmp/efs.$$ ] && cp /tmp/efs.$$ tw_bounds_ctg_l$L.csv
done
for L in 4 5 6 7; do hpc_ssh "cat ~/$R/qpu_iqm_ef_l$L.csv 2>/dev/null" > /tmp/efs.$$ && [ -s /tmp/efs.$$ ] && cp /tmp/efs.$$ qpu_iqm_ef_l$L.csv; done
rm -f /tmp/efs.$$
wc -l structure_grid_l[567].csv qpu_esp3_l[567].csv tw_bounds_grid_l[567].csv tw_bounds_ctg_l[567].csv qpu_iqm_ef_l*.csv 2>/dev/null
python3 adv_agg.py > /dev/null && python3 efig_calc.py | tail -16
python3 - <<'PY'
import json, math, os
d = json.load(open('efig.json'))
pend = []
have = {r['layers'] for r in json.load(open('adv.json'))['rows'] if any(c['o'] == 'ctg' for cs in (r.get('paths_ps') or []) for c in cs)}
ip = [f"{x['nq']} q" for x in d['series']['log2'] if x['cls_status'] == 'interpolated']
if ip: pend.append('interpolated until their circuits land: ' + ', '.join(ip))
if not d['meta']['iqm_route']: pend.append('IQM transpiles (Kingston overhead stands in)')
cs = d['meta'].get('cal_src', {})
for b_ in ('IonQ', 'IQM'):
    if not cs.get(b_, '').startswith('calibrated'): pend.append(f'{b_} noise calibration (modelled until then)')
if not os.path.exists('gpu_rep.csv'): pend.append('GPU run-to-run repeats')
d['meta']['pending'] = pend
clean = lambda x: None if isinstance(x, float) and not math.isfinite(x) else {k: clean(v) for k, v in x.items()} if isinstance(x, dict) else [clean(v) for v in x] if isinstance(x, list) else x
os.makedirs('efig_docs', exist_ok=True)
json.dump(clean(d['meta']), open('efig_docs/meta.json', 'w'), allow_nan=False)
for k, v in d['series'].items(): json.dump({'points': clean(v)}, open(f'efig_docs/{k}.json', 'w'), allow_nan=False)
print('pending:', pend)
PY
W=/cs/research/pplv/comp_bridge/qclip_wt/energyfig/analysis/bench/efig
scp -q efig.json efig_calc.py efig_plot.py tails-agent:$W/ && ssh tails-agent "cd $W && /cs/research/pplv/comp_bridge/.venv/bin/python efig_plot.py efig.json efig_log2 log2 && /cs/research/pplv/comp_bridge/.venv/bin/python efig_plot.py efig.json efig_L4 L4 && /cs/research/pplv/comp_bridge/.venv/bin/python efig_plot.py efig.json efig_L8 L8 && /cs/research/pplv/comp_bridge/.venv/bin/python efig_plot.py efig.json efig_log2_supp log2 with_breakeven" && scp -q "tails-agent:$W/efig_*.p*" . && ls -la efig_*.pdf
# paper files: LaTeX figure + methods, and the page's 'paper' store document (push it with ArtifactData set, pinned to its version)
python3 efig_setup_html.py && python3 efig_tex.py && python3 efig_paperdoc.py "$(ssh tails-agent 'cd /cs/research/pplv/comp_bridge/qclip_wt/energyfig && git rev-parse --short HEAD')"

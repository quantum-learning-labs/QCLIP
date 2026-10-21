# QCLIP Readout Atlas (2026-10-21): figures of the CPTP report + the geometry run (DLA, Fisher rank, Renyi entropy, magic).
# usage: python3 analysis/cptp/atlas/build.py <geo_dir>   (geo_dir = results/geometry/<run>/ JSONs from geometry.py, merged; on the MSI: tools/viz/geo)
# writes viz/index.html from viz/template.html + viz/report_data.json
import json, sys, glob, os, datetime, statistics as st
V = os.path.dirname(os.path.abspath(__file__))
GEO_DIR = sys.argv[1]
report = json.load(open(f'{V}/report_data.json'))

FAMILIES = [   # (key, label, colour token, file-name prefix)
    ('init', 'Training init', 'var(--ink-3)', 'init'),
    ('ps', 'Post-selection', 'var(--s1)', 'hp_hp2_nodiscard_tl3'),
    ('plain', 'Discard, plain', 'var(--s2)', 'hp_hp2_discard_tl3_pw0'),
    ('ws', 'Discard, warm start', 'var(--s3)', 'hp_do1_discard_tl3_pw0_initnd'),
    ('kd', 'Discard, distilled', 'var(--s4)', 'hp_kd1_discard_kd_ws'),
]
recs = {}
for f in sorted(glob.glob(f'{GEO_DIR}/*.json')):
    name = os.path.basename(f)[:-5]
    if name in ('dla', 'reference'): continue
    fam = next(k for k, _, _, p in FAMILIES if name.startswith(p))
    recs.setdefault(fam, []).append(json.load(open(f)))
dla = json.load(open(f'{GEO_DIR}/dla.json')); ref = json.load(open(f'{GEO_DIR}/reference.json'))
med = lambda a: st.median(a)
def pool(fam, key): return [s[key] for r in recs.get(fam, []) for s in r['sentences']]
def frac(fam, k): return [s[f'rank_{k}'] / s[f'max_{k}'] for r in recs.get(fam, []) for s in r['sentences'] if s[f'max_{k}']]
fams = [(k, l, c) for k, l, c, _ in FAMILIES if k in recs]
nseeds = {k: len(recs[k]) for k, _, _ in fams}
S0 = recs['ps'][0]['sentences']
nsent = len(S0)

models = [dict(label=l + (f' ({nseeds[k]} seeds)' if k != 'init' else ''), color=c, frac_full=frac(k, 'full'), frac_ps=frac(k, 'ps'), frac_tr=frac(k, 'tr'))
          for k, l, c in fams]

# ---- DLA ----
boxes = dla['boxes']; strs = [s for s in dla['structures'] if s.get('dim')]
w = sum(s['count'] for s in strs)
n_full = sum(s['count'] for s in strs if s['dim'] == s['full'])
cov = sorted((s['params'] / s['dim'], s['count']) for s in strs)
cum, cov_med = 0, None
for c, n in cov:
    cum += n
    if cum >= w / 2: cov_med = c; break
box_full = all(b['dim'] == b['full'] for b in boxes if b['N'] >= 2)
P_med = med(pool('ps', 'P'))
dla_out = dict(boxes=boxes, structures=strs,
    headline=('Every word box and every sentence circuit generates the full su(2ⁿ); the angles cover a sliver of it' if box_full and n_full == w
              else f'{n_full / w:.0%} of validation sentences generate the full su(2ⁿ); the angles cover a sliver of it'),
    caption=(f"Lie closure of the circuit's Pauli generators (fixed H and CX gates pushed to the front). Left: one CustomV5 word box on N wires, 3 layers "
             f"(1 and 2 layers give the same dimensions). Right: each distinct sentence-circuit structure in the validation set (dot size: how many sentences use it); "
             f"{n_full}/{w} validation sentences ({n_full / w:.0%}) reach the full algebra. The median sentence has {P_med:.0f} angles, "
             f"{cov_med:.2%} of its algebra's dimension, so the deep-circuit barren-plateau and overparametrisation results (which need angles ≳ dim g) do not apply here."))

# ---- Fisher rank ----
rk = {k: dict(full=med(frac(k, 'full')), ps=med(frac(k, 'ps')), tr=med(frac(k, 'tr'))) for k, _, _ in fams}
ds = {k: (sum(r['dataset_ps']['rank'] for r in recs[k]) / len(recs[k]), sum(r['dataset_tr']['rank'] for r in recs[k]) / len(recs[k]),
          sum(r['dataset_ps']['n_params'] for r in recs[k]) / len(recs[k])) for k, _, _ in fams}
qfi_out = dict(
    headline=f"Trained circuits move their full state in {rk['ps']['full']:.0%} of the possible directions; the post-selected readout keeps up to 14 of them, the trace up to 63",
    caption=(f"Rank of the quantum Fisher information w.r.t. the sentence's own angles, divided by its ceiling: min(P, 2ⁿ⁺¹ − 2) for the full pure state, "
             f"min(P, 14) for the normalised post-selected state |φ⟩ (a pure 3-qubit state has 14 real directions), min(P, 63) for the traced ρ (SLD Fisher). "
             f"{nsent} validation sentences per checkpoint, all checkpoints compiled in discard mode so both readouts come from the same weights. "
             f"Dataset level (all {nsent} sentences stacked, ~{ds['ps'][2]:.0f} angles): rank {ds['ps'][0]:.0f} for |φ⟩ and {ds['ps'][1]:.0f} for ρ in the post-selection checkpoints, "
             f"{ds['plain'][0]:.0f} / {ds['plain'][1]:.0f} in plain discard."))

# ---- spectrum of one typical sentence (median n, median P), post-selection seed 0 vs plain discard seed 0 ----
nmed = med([s['n'] for s in S0])
cands = [i for i, s in enumerate(S0) if s['n'] == nmed]
i0 = sorted(cands, key=lambda i: abs(S0[i]['P'] - P_med))[0]
def spec(fam, key):
    s = recs[fam][0]['sentences'][i0]; ev = sorted((max(v, 0) for v in s[key]), reverse=True); m = ev[0] or 1
    return [v / m for v in ev]
sp = S0[i0]
spectrum = dict(P=sp['P'], xt=[1] + [x for x in (14, 28, 42, 56, 63, 78, 90) if x <= sp['P']],
    series=[dict(label='full state (post-selection ckpt)', y=spec('ps', 'qfi_full'), color='var(--ink-3)', dash=True),
            dict(label='post-selected |φ⟩ (post-selection ckpt)', y=spec('ps', 'qfi_ps'), color='var(--s1)'),
            dict(label='traced ρ (plain discard ckpt)', y=spec('plain', 'qfi_tr'), color='var(--s2)')],
    marks=[dict(at=14, label='14'), dict(at=63, label='63')] if sp['P'] > 63 else [dict(at=14, label='14')],
    headline=f"One sentence, {sp['P']} angles: the readouts cut the Fisher spectrum off at 14 and 63 directions",
    caption=(f"Sorted QFI eigenvalues for “{sp['sentence']}” ({sp['n']} qubits), normalised to the largest. "
             f"Ranks: full state {recs['ps'][0]['sentences'][i0]['rank_full']}, |φ⟩ {recs['ps'][0]['sentences'][i0]['rank_ps']}, "
             f"ρ {recs['plain'][0]['sentences'][i0]['rank_tr']} (numerical rank, relative cut 1e-9)."))

# ---- entropy ----
def s2(fam): return pool(fam, 'S2')
ent_rows = [dict(label=l, vals=s2(k), color=c) for k, l, c in fams]
ent_rows.append(dict(label='Haar-random 7-qubit state, traced', vals=ref['m4']['S2'], color='var(--ink-3)', hollow=True))
entropy = dict(rows=ent_rows, legend=[dict(label='median (dot), IQR (bar), 5–95% (line)', color='var(--ink-3)', kind='dot')],
    headline=f"Traced sentence states carry {med(s2('ps')):.1f}–{med(s2('plain')):.1f} bits of S–E entanglement, close to a random state's {med(ref['m4']['S2']):.1f}",
    caption=(f"Rényi-2 entropy S₂ = −log₂ Tr ρ² of the 3-qubit sentence state after tracing E (= the S–E Rényi-2 entanglement, as the full state is pure). "
             f"For post-selection-trained weights this is what tracing would give; their post-selected |φ⟩ is pure (S₂ = 0). "
             f"Reference: Haar-random states on 3 + 4 qubits traced to 3. Medians: init {med(s2('init')):.2f}, post-selection {med(s2('ps')):.2f}, plain discard {med(s2('plain')):.2f}, "
             f"warm start {med(s2('ws')):.2f}, distilled {med(s2('kd')):.2f} bits."))

# ---- magic ----
mg_rows = []
for k, l, c in fams:
    mg_rows.append(dict(label=f'{l}: |φ⟩', vals=pool(k, 'M2_ps'), color=c, hollow=True))
    mg_rows.append(dict(label='… traced ρ (mixed M̃₂)', vals=pool(k, 'M2_tr'), color=c, sub=True))
mg_rows.append(dict(label='Haar-random 3-qubit pure state', vals=ref['m0']['M2'], color='var(--ink-3)', hollow=True))
mg_rows.append(dict(label='Haar 3+4 qubits, traced (M̃₂)', vals=ref['m4']['M2'], color='var(--ink-3)', sub=True))
mps = {k: med(pool(k, 'M2_ps')) for k, _, _ in fams}; mtr = {k: med(pool(k, 'M2_tr')) for k, _, _ in fams}
magic = dict(rows=mg_rows, refs=[dict(v=med(ref['m0']['M2']), label='Haar')],
    legend=[dict(label='pure post-selected |φ⟩ (hollow)', color='var(--ink-3)', kind='dot'), dict(label='traced ρ (filled)', color='var(--ink-2)', kind='dot')],
    headline='MAGIC_HEADLINE',
    caption=(f"M₂ = −log₂(Σ_P ⟨P⟩⁴ / 8) over the 64 Pauli strings for the pure |φ⟩; for ρ the mixed version M̃₂ = −log₂(Σ_P Tr(ρP)⁴ / Σ_P Tr(ρP)²), "
             f"which equals M₂ on pure states and is 0 for I/8 (it is not a magic monotone for mixed states). Stabiliser states have M₂ = 0. "
             f"Medians |φ⟩ / ρ: init {mps['init']:.2f} / {mtr['init']:.2f}, post-selection {mps['ps']:.2f} / {mtr['ps']:.2f}, plain discard {mps['plain']:.2f} / {mtr['plain']:.2f}, "
             f"warm start {mps['ws']:.2f} / {mtr['ws']:.2f}, distilled {mps['kd']:.2f} / {mtr['kd']:.2f} bits; Haar pure {med(ref['m0']['M2']):.2f}, traced {med(ref['m4']['M2']):.2f}."))

facts = [dict(v=f"{n_full / w:.0%}", k='of sentence circuits generate the full su(2ⁿ)'),
         dict(v=f"{P_med:.0f}", k=f'angles in the median sentence, {cov_med:.2%} of dim g'),
         dict(v=f"{rk['ps']['tr']:.0%} / {rk['plain']['tr']:.0%}", k='median Fisher rank of ρ / ceiling: post-sel. vs discard ckpts'),
         dict(v=f"{med(s2('plain')):.2f} bits", k='median S₂ of a plain-discard sentence state (max 3)')]
GEO = dict(models=models, dla=dla_out, qfi=qfi_out, spectrum=spectrum, entropy=entropy, magic=magic, facts=facts)
# hand-written headlines, after looking at the numbers (2026-10-21)
fr = med(pool('ps', 'rank_full'))
qfi_out['headline'] = f"Half of each sentence's angles are redundant: Fisher rank {fr:.0f} for {P_med:.0f} angles, the same for every training method"
qfi_out['caption'] += (f" The full-state rank ({fr:.0f} for the typical 7-qubit sentence, whether it has 108 or 117 distinct angles) also caps the traced state's: "
                       f"ρ reaches {med(pool('ps', 'rank_tr')):.0f} of 63, so the trace ceiling is set by the circuit, not by the 3-qubit state space. Training init: {med(pool('init', 'rank_full')):.0f} / {med(pool('init', 'rank_tr')):.0f}.")
entropy['headline'] = f"Trained sentence states are as entangled with E as random states, about {med(s2('ps')):.1f} of 3 bits, whichever readout trained them"
magic['headline'] = (f"Training takes the near-Clifford init to near-random magic, and discard states are not less magical (conjecture 4b refuted)")
facts[2] = dict(v=f"{fr:.0f} / {P_med:.0f}", k='Fisher rank of the full state / angles, every model')
facts[3] = dict(v=f"{med(s2('plain')):.2f} bits", k=f"median S₂ of a plain-discard sentence state (random state: {med(ref['m4']['S2']):.2f}, max 3)")
summary = {k: dict(seeds=nseeds[k], rank=rk[k], dataset=ds[k], S2=med(s2(k)), M2_ps=mps[k], M2_tr=mtr[k], P=med(pool(k, 'P')), p0=med(pool(k, 'p0'))) for k, _, _ in fams}
json.dump(dict(summary=summary, dla_full_share=n_full / w, cov_median=cov_med, haar=dict(S2_m4=med(ref['m4']['S2']), M2_m0=med(ref['m0']['M2']), M2_m4=med(ref['m4']['M2']))),
          open(f'{V}/geo_summary.json', 'w'), indent=1)
html = open(f'{V}/template.html').read()
html = html.replace('__REPORT__', json.dumps(report, ensure_ascii=False)).replace('__GEO__', json.dumps(GEO, ensure_ascii=False))
html = html.replace('__GEO_SRC__', 'qclip branch geometry, analysis/cptp/followup/geometry.py').replace('__BUILT__', datetime.date.today().isoformat())
open(f'{V}/index.html', 'w').write(html)
print(json.dumps(summary, indent=1)); print('wrote', f'{V}/index.html', len(html) // 1024, 'KB')

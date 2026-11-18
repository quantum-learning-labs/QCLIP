# Paper figure from efig.json (efig_calc.py), 2x2: (a) break-even two-qubit error vs qubits (vs best classical, and vs GPU statevector
# only), (b-d) energy per decision per QPU backend (today / 10x / 100x lower 2q error) against best classical and GPU statevector.
# usage: python efig_plot.py efig.json out_prefix [rule]   (rule: log2 | L4 | L8; default log2) -> out_prefix.pdf / .png
import json, sys, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = json.load(open(sys.argv[1])); out = sys.argv[2]; rule = sys.argv[3] if len(sys.argv) > 3 else 'log2'
P, M = d['series'][rule], d['meta']
C = dict(IBM='#2f62d6', IonQ='#c2410c', IQM='#0f8a6a', cls='#1d2230', sv='#8a5cd1')
LS = {'1': '-', '0.1': (0, (6, 3)), '0.01': (0, (1.5, 2.5))}
BK = ['IBM', 'IonQ', 'IQM']; NAME = dict(IBM='IBM Kingston', IonQ='IonQ Forte', IQM='IQM Garnet-class')
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(2, 2, figsize=(7.1, 5.6), constrained_layout=True)
a, panels = ax[0, 0], [ax[0, 1], ax[1, 0], ax[1, 1]]

def cross(key, base):
    B = (lambda p: p['sv']['J'] if p.get('sv') else None) if base == 'sv' else (lambda p: p['classical'])
    for i, p in enumerate(P):
        q, c = p['qpu'][key[0]].get(key[1]), B(p)
        if q is None or c is None: continue
        if q <= c:
            if i == 0: return p['nq']
            o = P[i - 1]; q0, c0 = o['qpu'][key[0]][key[1]], B(o)
            if c0 is None: return p['nq']
            d0, d1 = math.log10(q0) - math.log10(c0), math.log10(q) - math.log10(c)
            return o['nq'] + (p['nq'] - o['nq']) * d0 / (d0 - d1)
    return None

# (a) break-even error
for bk in BK:
    pts = [p for p in P if p['be'][bk] is not None]
    a.plot([p['nq'] for p in pts], [p['be'][bk] for p in pts], color=C[bk], lw=1.5, marker='o', ms=3, mfc='white', zorder=3)
    pts = [p for p in P if p.get('be_sv', {}).get(bk) is not None]
    a.plot([p['nq'] for p in pts], [p['be_sv'][bk] for p in pts], color=C[bk], lw=1.0, ls=(0, (3, 2)), marker='^', ms=3, mfc='white', zorder=3)
    for m in ('1', '0.1', '0.01'):
        a.axhline(M['today'][bk] * float(m), color=C[bk], lw=0.7, ls=LS[m], alpha=0.8 if m == '1' else 0.55, zorder=1)
a.set_yscale('log'); a.set_xlim(14, 128); a.set_ylim(1e-6, 3e-2)
a.set_xlabel('total qubits'); a.set_ylabel('break-even two-qubit error (effective)')
a.set_title('(a) QPU wins below its curve', fontsize=9, loc='left')

# (b-d) energy per backend, shared axes
vals = [p['classical'] for p in P] + [p['sv']['J'] for p in P if p.get('sv') and p['sv']['status'] != 'extrapolated'] + \
       [p['qpu'][b][m] for p in P for b in BK for m in ('0.1', '0.01') if p['qpu'][b].get(m)]
ylo, yhi = 10 ** (math.floor(math.log10(min(vals))) - 1), 10 ** (math.ceil(math.log10(max(vals))) + 1)
for k, (bk, axp) in enumerate(zip(BK, panels)):
    sv = [p for p in P if p.get('sv')]
    xm = [p for p in sv if p['sv']['status'] != 'extrapolated']; xe = [p for p in sv if p['sv']['status'] == 'extrapolated']
    axp.plot([p['nq'] for p in xm], [p['sv']['J'] for p in xm], color=C['sv'], lw=1.3, zorder=3)
    if xe: axp.plot([xm[-1]['nq']] + [p['nq'] for p in xe], [xm[-1]['sv']['J']] + [p['sv']['J'] for p in xe], color=C['sv'], lw=1.3, ls=(0, (5, 3)), zorder=3)
    for p in xm: axp.plot([p['nq']], [p['sv']['J']], marker='o', ms=3, color=C['sv'], mfc=C['sv'] if p['sv']['status'] == 'measured' else 'white', zorder=4)
    axp.plot([p['nq'] for p in P], [p['classical'] for p in P], color=C['cls'], lw=2.0, zorder=5)
    for p in P:
        if p['cls_status'] == 'measured': axp.plot([p['nq']], [p['classical']], 'o', ms=3.5, color=C['cls'], zorder=6)
    nb = nt = 0
    for m in ('1', '0.1', '0.01'):
        pts = [p for p in P if p['qpu'][bk].get(m) is not None]
        axp.plot([p['nq'] for p in pts], [p['qpu'][bk][m] for p in pts], color=C[bk], lw=1.4 if m == '1' else 1.2, ls=LS[m], zorder=2)
        x = cross((bk, m), 'cls')
        if x: nb += 1; axp.annotate(f'{round(x)} q', (x, ylo), xytext=(0, 3 + 9 * nb), textcoords='offset points', ha='center', fontsize=6.5, color=C[bk],
                           arrowprops=dict(arrowstyle='-', color=C[bk], lw=0.8))
        x = cross((bk, m), 'sv')
        if x: nt += 1; axp.annotate(f'sv {round(x)}', (x, yhi), xytext=(0, -3 - 9 * nt), textcoords='offset points', ha='center', fontsize=6.5, color=C['sv'],
                           arrowprops=dict(arrowstyle='-', color=C[bk], lw=0.8))
    axp.set_yscale('log'); axp.set_xlim(14, 128); axp.set_ylim(ylo, yhi)
    axp.set_xlabel('total qubits'); axp.set_ylabel('energy per decision (J)')
    axp.set_title(f'({"bcd"[k]}) {NAME[bk]}', fontsize=9, loc='left', color=C[bk])
for x in [a] + panels: x.grid(True, which='major', lw=0.3, color='#d9dce2')

h = [Line2D([], [], color=C['cls'], lw=2, label='best classical (GPU)'), Line2D([], [], color=C['sv'], lw=1.3, label='GPU statevector'),
     Line2D([], [], color=C['sv'], lw=1.3, ls=(0, (5, 3)), label='statevector, exp. fit')] + \
    [Line2D([], [], color='0.35', lw=1, ls=LS[m], label=l) for m, l in (('1', 'QPU today'), ('0.1', '10× lower 2q error'), ('0.01', '100× lower'))] + \
    [Line2D([], [], color='0.3', lw=1.2, marker='o', mfc='white', ms=3, label='(a) vs best classical'), Line2D([], [], color='0.3', lw=1, ls=(0, (3, 2)), marker='^', mfc='white', ms=3, label='(a) vs statevector only')]
fig.legend(handles=h, loc='outside lower center', ncol=4, frameon=False, fontsize=7.2)
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
print('wrote', out + '.pdf')

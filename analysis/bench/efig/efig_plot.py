# Paper figure from efig.json (efig_calc.py): (a) break-even two-qubit error vs qubits, (b) energy per decision vs qubits.
# usage: python efig_plot.py efig.json out_prefix [rule]   (rule: log2 | L4 | L8; default log2) -> out_prefix.pdf / .png
import json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

d = json.load(open(sys.argv[1])); out = sys.argv[2]; rule = sys.argv[3] if len(sys.argv) > 3 else 'log2'
P, M = d['series'][rule], d['meta']
C = dict(IBM='#2f62d6', IonQ='#c2410c', IQM='#0f8a6a', cls='#1d2230', sv='#8a5cd1')
LS = {'1': '-', '0.1': (0, (6, 3)), '0.01': (0, (1.5, 2.5))}
BK = ['IBM', 'IonQ', 'IQM']
plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)

def mark(ax, xs, ys, sts, col, z=3):
    for x, y, s in zip(xs, ys, sts):
        ax.plot([x], [y], marker='D' if s == 'interpolated' else 'o', ms=3.6 if s == 'interpolated' else 4,
                mfc=col if s == 'measured' else 'white', mec=col, mew=0.9, zorder=z)

# (a) break-even
for bk in BK:
    pts = [p for p in P if p['be'][bk] is not None]
    a.plot([p['nq'] for p in pts], [p['be'][bk] for p in pts], color=C[bk], lw=1.4, zorder=3)
    mark(a, [p['nq'] for p in pts], [p['be'][bk] for p in pts], [p['cls_status'] for p in pts], C[bk])
    for m in ('1', '0.1', '0.01'):
        a.axhline(M['today'][bk] * float(m), color=C[bk], lw=0.8, ls=LS[m], alpha=0.85 if m == '1' else 0.6, zorder=1)
a.set_yscale('log'); a.set_xlim(14, 128); a.set_ylim(1e-6, 3e-2)
a.set_xlabel('total qubits'); a.set_ylabel('break-even two-qubit error (effective)')
a.set_title('(a) QPU wins below its curve', fontsize=9, loc='left')

# (b) energy
mark(b, [p['nq'] for p in P], [p['classical'] for p in P], [p['cls_status'] for p in P], C['cls'], z=5)
b.plot([p['nq'] for p in P], [p['classical'] for p in P], color=C['cls'], lw=2.2, zorder=4)
for bk in BK:
    for m in ('1', '0.1', '0.01'):
        pts = [p for p in P if p['qpu'][bk].get(m) is not None]
        b.plot([p['nq'] for p in pts], [p['qpu'][bk][m] for p in pts], color=C[bk], lw=1.2 if m != '1' else 1.5, ls=LS[m], zorder=2)
b.set_yscale('log'); b.set_xlim(14, 128); b.set_ylim(1e-1, 1e12)
b.set_xlabel('total qubits'); b.set_ylabel('energy per decision (J)')
b.set_title('(b) energy per decision', fontsize=9, loc='left')
for ax in (a, b): ax.grid(True, which='major', lw=0.3, color='#d9dce2')

h = [Line2D([], [], color=C['cls'], lw=2.2, label='best classical (GPU)')] + [Line2D([], [], color=C[k], lw=1.5, label=n) for k, n in
     (('IBM', 'IBM Kingston'), ('IonQ', 'IonQ Forte'), ('IQM', 'IQM Garnet-class'))] + \
    [Line2D([], [], color='0.35', lw=1, ls=LS[m], label=l) for m, l in (('1', 'today'), ('0.1', '10× lower 2q error'), ('0.01', '100× lower'))] + \
    [Line2D([], [], color='0.2', marker='o', mfc='0.2', ls='', ms=4, label='measured'), Line2D([], [], color='0.2', marker='o', mfc='white', ls='', ms=4, label='path upper bound'),
     Line2D([], [], color='0.2', marker='D', mfc='white', ls='', ms=3.6, label='interpolated depth')]
fig.legend(handles=h, loc='outside lower center', ncol=5, frameon=False, fontsize=7.5)
fig.savefig(out + '.pdf'); fig.savefig(out + '.png', dpi=200)
print('wrote', out + '.pdf')

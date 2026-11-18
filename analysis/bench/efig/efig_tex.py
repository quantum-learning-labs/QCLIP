# LaTeX for the paper: efig_figure.tex (figure environment + caption, numbers from efig.json) and efig_methods.tex (from efig_setup.md).
# usage: python3 efig_tex.py   (in adv_data; writes efig_figure.tex, efig_methods.tex)
import json, re
SUP = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁻', '0123456789-')); SUB = dict(zip('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
SYM = {'×': r'$\times$', '≤': r'$\le$', '≥': r'$\ge$', 'ε': r'$\varepsilon$', 'χ': r'$\chi$', 'η': r'$\eta$', 'σ': r'$\sigma$', '·': r'$\cdot$',
       'µ': r'$\mu$', '−': '--', '–': '--', '—': '---', '∈': r'$\in$', '→': r'$\to$', '“': '``', '”': "''", '≈': r'$\approx$', '₂': r'$_2$'}
def tex(t):
    links = []
    def keep(m): links.append((m.group(1), m.group(2))); return f'\x00{len(links) - 1}\x00'
    t = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', keep, t)
    codes = []
    def keepc(m): codes.append(m.group(1)); return f'\x01{len(codes) - 1}\x01'
    t = re.sub(r'`([^`]+)`', keepc, t)
    t = t.replace('\\', r'\textbackslash{}')
    for a, b in (('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\textasciitilde{}'), ('^', r'\^{}')):
        t = t.replace(a, b)
    t = re.sub(r'\*\*([^*]+)\*\*', r'\\textbf{\1}', t)
    t = re.sub('[' + ''.join(SUP) + ']+', lambda m: '$^{' + ''.join(SUP[c] for c in m.group(0)) + '}$', t)
    t = re.sub('[' + ''.join(SUB) + ']+', lambda m: '$_{' + ''.join(SUB[c] for c in m.group(0)) + '}$', t)
    t = ''.join(SYM.get(c, c) for c in t)
    t = t.replace('$$', '')
    esc = lambda x: x.replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%').replace('#', r'\#').replace('{', r'\{').replace('}', r'\}')
    t = re.sub('\x01(\\d+)\x01', lambda m: r'\texttt{' + esc(codes[int(m.group(1))]) + '}', t)
    t = re.sub('\x00(\\d+)\x00', lambda m: r'\href{' + links[int(m.group(1))][1] + '}{' + tex(links[int(m.group(1))][0]) + '}', t)
    return t
out, depth = [], 0
def setdepth(d):
    global depth
    while depth < d: out.append('  ' * depth + r'\begin{itemize}'); depth += 1
    while depth > d: depth -= 1; out.append('  ' * depth + r'\end{itemize}')
for line in open('efig_setup.md').read().split('\n'):
    if line.startswith('## '): setdepth(0); out.append(r'\subsection*{' + tex(line[3:]) + '}\\label{sec:energy-setup}'); continue
    if line.startswith('### '): setdepth(0); out.append(r'\paragraph{' + tex(line[4:]) + '}'); continue
    m = re.match(r'^( *)- (.*)$', line)
    if m: setdepth(len(m.group(1)) // 2 + 1); out.append('  ' * depth + r'\item ' + tex(m.group(2))); continue
    m = re.match(r'^ *(\d+)\. (.*)$', line)
    if m: setdepth(0); out.append(f'({m.group(1)}) ' + tex(m.group(2)) + r'\\'); continue
    if line.strip(): setdepth(0); out.append(tex(line))
    else: out.append('')
setdepth(0)
open('efig_methods.tex', 'w').write('% Methods: circuits, scaling and experimental setup of the energy figure (generated from efig_setup.md by efig_tex.py).\n'
                                     '% Needs \\usepackage{hyperref}.\n' + '\n'.join(out) + '\n')
d = json.load(open('efig.json')); M = d['meta']; cr = M.get('cross', {}).get('log2', {})
def x(b, m, base):
    c = cr.get(f'{b}|{m}|{base}')
    if not c or c.get('x') is None: return 'none to 124'
    xx = round(c['x'])
    if c.get('lo') is None: return f'{xx}'
    lo, hi = round(min(c['x'], c['lo'])), round(max(c['x'], c['hi']))
    return f'{xx}' if lo == hi else f'{xx} [{lo}--{hi}]'
B = [('IBM', 'IBM Kingston'), ('IonQ', 'IonQ Forte'), ('IQM', 'IQM Garnet-class')]
interp = sorted({p['L'] for p in d['series']['log2'] if p['cls_status'] == 'interpolated'})
cap = (r'\textbf{Inference energy per decision: QPUs against the best classical method.} '
       r'Energy (whole system) for one image-choice decision of the CCG-structured post-selected circuit family (SVO-Probes standard test set), grown in '
       r'qubits per wire with depth $L=\mathrm{round}(\log_2 n_q)$, on (a) IBM Kingston, (b) IonQ Forte and (c) an IQM Garnet-class device, at today\textquoteright s '
       r'two-qubit error (solid) and 10$\times$ (dashed) and 100$\times$ (dotted) lower. '
       r'Black: best classical method, the cheapest of GPU statevector, exact tensor-network contraction, tensor-network sampler and MPS, with path search '
       r'amortised (filled markers measured, otherwise best-path upper bounds); purple: GPU statevector simulation alone (dashed: exponential extrapolation). '
       r'Shaded: 16th--84th percentile of ' + str(M.get('unc', {}).get('draws', '')) + r' Monte Carlo draws over measurement, path-quality, interpolation, '
       r'calibration, power and timing uncertainty. Classical bands are one-sided, as a better contraction path may exist; they widen with circuit size '
       r'as the path search improves less reliably, and vanish where a fixed per-run overhead, not the path, sets the cost. '
       "Bottom ticks mark where a QPU line drops below the best classical method, top ticks (``sv'') where it drops below statevector simulation, with intervals. "
       + 'Against the best classical method, at 10$\\times$ / 100$\\times$ lower two-qubit error: '
       + '; '.join(f'{n} {x(b, "0.1", "cls")} / {x(b, "0.01", "cls")} qubits' for b, n in B)
       + '; none at today\'s error up to 124 qubits. Against statevector simulation alone, at today\'s error: '
       + '; '.join(f'{n} {x(b, "1", "sv")}' for b, n in B) + ' qubits. '
       + (f'Depths {", ".join(map(str, interp))} are interpolated from 4- and 8-layer circuits (preliminary). ' if interp else '')
       + r'Setup and sources: Section~\ref{sec:energy-setup}.')
open('efig_figure.tex', 'w').write('% Energy figure (generated by efig_tex.py from efig.json, ' + M['updated'] + ')\n'
                                    '\\begin{figure*}[t]\n  \\centering\n  \\includegraphics[width=\\textwidth]{efig_log2.pdf}\n'
                                    '  \\caption{' + cap + '}\n  \\label{fig:energy}\n\\end{figure*}\n')
print('ok')

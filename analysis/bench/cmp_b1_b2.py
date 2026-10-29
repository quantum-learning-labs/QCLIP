# are bench_b1 and bench_b2 the same logical circuits? exact distributions, readout infos, circuit names and gate counts
import pickle, sys
from qiskit import qpy
for s in sys.argv[1:]:
    r1, r2 = (pickle.load(open(f'bench_{b}/{s}/raw.pkl', 'rb')) for b in ('b1', 'b2'))
    p1, p2 = r1['exact']['probs'], r2['exact']['probs']
    md = max(abs(p1[j].get(k, 0) - p2[j].get(k, 0)) for j in range(len(p1)) for k in set(p1[j]) | set(p2[j]))
    c1, c2 = (qpy.load(open(f'bench_{b}/{s}/logical.qpy', 'rb')) for b in ('b1', 'b2'))
    ops = all(a.count_ops() == b.count_ops() and a.num_qubits == b.num_qubits for a, b in zip(c1, c2))
    print(s, 'n', len(p1), len(p2), 'infos equal', r1['infos'] == r2['infos'], 'max |dp| exact', f'{md:.2e}', 'same gate counts', ops,
          'names', c1[0].name, c2[0].name)

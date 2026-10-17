import numpy as np, statistics as st
from qiskit_ibm_runtime import fake_provider as fp
names = [n for n in dir(fp) if n.startswith('Fake') and not n.endswith('V1') and 'Provider' not in n]
print('available:', len(names), [n for n in names if n in ('FakeMiami','FakeTorino','FakeFez','FakeMarrakesh','FakeSherbrooke','FakeKyiv','FakeBrisbane','FakeQuebec')])
for n in ('FakeMiami', 'FakeTorino', 'FakeFez', 'FakeSherbrooke'):
    if not hasattr(fp, n): continue
    b = getattr(fp, n)(); t = b.target
    twoq = [g for g in t.operation_names if t.operation_from_name(g).num_qubits == 2 and g not in ('swap',)]
    e2 = [p.error for g in twoq for q, p in (t[g] or {}).items() if p is not None and p.error is not None]
    d2 = [p.duration for g in twoq for q, p in (t[g] or {}).items() if p is not None and p.duration]
    e1 = [p.error for g in ('sx',) for q, p in (t[g] or {}).items() if p is not None and p.error is not None]
    ro = [p.error for q, p in (t['measure'] or {}).items() if p is not None and p.error is not None]
    T1 = [q.t1 for q in t.qubit_properties if q.t1]; T2 = [q.t2 for q in t.qubit_properties if q.t2]
    print(f"{n}: {b.num_qubits} q, 2q gates {twoq}, median 2q err {st.median(e2):.4f}, 2q dur {st.median(d2)*1e9:.0f} ns, "
          f"median sx err {st.median(e1):.5f}, median readout err {st.median(ro):.4f}, median T1 {st.median(T1)*1e6:.0f} us, T2 {st.median(T2)*1e6:.0f} us, dt {b.dt}")

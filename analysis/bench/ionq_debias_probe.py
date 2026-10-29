# Does the IonQ cloud SIMULATOR (forte-1 noise) apply debiasing? Same circuits with/without, Hellinger to exact.
import os, pickle, inspect
from qiskit import qpy, transpile
import qiskit_ionq
from qiskit_ionq import IonQProvider, ErrorMitigation
import bench_lib as B
d = 'results/bench_smoke/ps_swap1'
raw = pickle.load(open(d + '/raw.pkl', 'rb')); circs = qpy.load(open(d + '/logical.qpy', 'rb'))[:4]
sim = IonQProvider(token=open(os.path.expanduser('~/.ionq_key')).read().strip()).get_backend('simulator')
assert 'qpu' not in sim.name.lower()
tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=0)
print('ErrorMitigation', list(ErrorMitigation))
for tag, kw in [('plain', {}), ('debias', {'error_mitigation': ErrorMitigation.DEBIASING})]:
    job = sim.run(tc, shots=8192, noise_model='forte-1', **kw)
    res = job.result()
    for sharpen in ([False, True] if tag == 'debias' else [False]):
        try:
            cnt = job.get_counts(sharpen=sharpen) if sharpen else res.get_counts()
        except Exception as e:
            print(tag, 'sharpen', sharpen, 'ERR', e); continue
        cnt = cnt if isinstance(cnt, list) else [cnt]
        print(tag, 'sharpen' if sharpen else 'avg', [round(B.hellinger(c, raw['exact']['probs'][j]), 4) for j, c in enumerate(cnt)], 'nkeys', [len(c) for c in cnt])
    try: print(tag, 'job meta', {k: v for k, v in (job._metadata or {}).items()} if hasattr(job, '_metadata') else '', str(getattr(job, '_job', ''))[:300])
    except Exception as e: print('meta err', e)
print(inspect.signature(type(job).get_counts))

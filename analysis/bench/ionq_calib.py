# IonQ cloud SIMULATOR (noise model forte-1) run of the calibration circuits dumped by shotcost_noise.py --dump: writes the same npz as
# the Aer runs (P, U, k, idx) so the like-for-like F_eff fit treats every backend alike. Never a QPU: the backend is the simulator and is
# checked by name before anything is sent. The API key is read at runtime from $IONQ_API_KEY or ~/.ionq_key and never printed or stored.
#   PYTHONPATH=.:analysis/bench python analysis/bench/ionq_calib.py <dump prefix> <out.npz> [--shots 4096] [--batch 50] [--pairs N]
import argparse, os, pickle, time, json
import numpy as np
from qiskit import qpy, transpile
from qiskit_ionq import IonQProvider
from shotcost_noise import stats, counts_to_probs, K

ap = argparse.ArgumentParser()
ap.add_argument('dump'); ap.add_argument('out')
ap.add_argument('--noise', default='forte-1'); ap.add_argument('--shots', type=int, default=4096)
ap.add_argument('--batch', type=int, default=50); ap.add_argument('--pairs', type=int, default=None)
a = ap.parse_args()
if os.path.exists(a.out): raise SystemExit(f'{a.out} exists')
meta = pickle.load(open(a.dump + '.pkl', 'rb'))
circs = qpy.load(open(a.dump + '.qpy', 'rb'))
stride, rows, meta_c = meta['stride'], meta['rows'], meta['meta_c']
if a.pairs: rows, meta_c, circs = rows[:a.pairs], meta_c[:a.pairs], circs[:a.pairs * stride]
key = os.environ.get('IONQ_API_KEY') or open(os.path.expanduser('~/.ionq_key')).read().strip()
sim = IonQProvider(token=key).get_backend('simulator')
del key
assert sim.name.lower() in ('ionq_simulator', 'simulator') and 'qpu' not in sim.name.lower(), f'refusing non-simulator backend {sim.name}'
tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=0)
t0 = time.time()
jobs = [sim.run(tc[i:i + a.batch], shots=a.shots, noise_model=a.noise) for i in range(0, len(tc), a.batch)]
open(a.out + '.jobs.txt', 'w').write(' '.join(j.job_id() for j in jobs) + '\n')
print(f'submitted {len(tc)} circuits in {len(jobs)} jobs ({time.time() - t0:.0f}s)', flush=True)
probs = []
for k, job in enumerate(jobs):
    cnt = job.result().get_counts()
    cnt = cnt if isinstance(cnt, list) else [cnt]
    for jj, c in enumerate(cnt):
        probs.append(counts_to_probs(dict(c), circs[k * a.batch + jj].num_clbits))
    print(f'job {k + 1}/{len(jobs)} done after {time.time() - t0:.0f}s', flush=True)
N = len(rows)
P, U, KK = np.zeros((N, 2, K + 1)), np.zeros((N, 2, K + 1)), np.zeros((N, 2), int)
for r, (n, oq, vals) in enumerate(meta_c):
    for j in range(2):
        P[r, j], U[r, j] = stats(probs[stride * r + j], oq, n); KK[r, j] = n - len(oq)
np.savez(a.out, P=P, U=U, k=KK, idx=np.array(rows))
json.dump(dict(checkpoint=meta['checkpoint'], profile='ionq_' + a.noise, pairs=N, shots=a.shots, wall_s=time.time() - t0,
               jobs=len(jobs)), open(a.out.replace('.npz', '') + '.json', 'w'), indent=1)
print('done', N, 'pairs', f'{time.time() - t0:.0f}s', flush=True)

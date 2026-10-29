# Linearity check: mean over many finite debias variants must equal the twirl-limit noise model's distribution.
# (a) original noise, (b) twirl-limit noise model, (c) average of V finite variants (twirl + readout flips, unflipped).
import copy, json, pickle, sys
from pathlib import Path
import numpy as np
from qiskit import qpy
import qiskit_ibm_runtime.fake_provider as fake_provider
from modules.utils.general import setup_exp, resolve_eval_config
from modules.compilation.quantum.emul import BackendManager
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
from qiskit.circuit import Delay
from qiskit_aer.noise import RelaxationNoisePass
import bench_lib as B
from debias import variants, unflip
from twirl_limit import twirled_noise_model
d = Path(sys.argv[1]); name = sys.argv[2]; N = int(sys.argv[3]); V = int(sys.argv[4]); SH = 200000
meta = json.load(open(d / 'summary.json'))['meta']; raw = pickle.load(open(d / 'raw.pkl', 'rb'))
circs = qpy.load(open(d / 'logical.qpy', 'rb'))[:N]; infos = raw['infos'][:N]; exact = raw['exact']['probs'][:N]
config, _, _ = setup_exp(resolve_eval_config(None, Path(meta['checkpoint'])), seed=0)
config['emulate'].update(set=meta['eval_set'], measurement_method=meta['meas'], noise=True, noise_profile='honest' if 'honest' in name else 'backend',
                         backend='density_matrix', aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False, 'max_parallel_threads': 8})
nb = BackendManager(config, meta['qlimit'], hardware_profile=fake_provider.FakeMiami())
idle = nb.honest['idle'] and 'honest' in name
nb.honest['idle'] = False; plain = nb.transpile(circs, batch_size=256, optimization_level=1); nb.honest['idle'] = idle
tgt = nb.hardware_profile.target
if idle:
    qp = tgt.qubit_properties
    t1s = [q.t1 if q and q.t1 else float('inf') for q in qp]; t2s = [min(q.t2, 1.99 * q.t1) if q and q.t1 and q.t2 else float('inf') for q in qp]
    sched = PassManager([ALAPScheduleAnalysis(target=tgt), PadDelay(target=tgt)])
    relax = PassManager([RelaxationNoisePass(t1s, t2s, dt=nb.hardware_profile.dt, op_types=[Delay])])
fin = (lambda c: relax.run(sched.run(c))) if idle else (lambda c: c)
nm0 = nb.backend.options.noise_model
def run(cs, nm):
    nb.backend.set_options(noise_model=nm, seed_simulator=11)
    res = nb.backend.run(cs, shots=SH).result(); return [dict(res.get_counts(j)) for j in range(len(cs))]
A = run([fin(c) for c in plain], nm0)
nmT, n2, nro = twirled_noise_model(nm0); Bt = run([fin(c) for c in plain], nmT)
rng = np.random.default_rng(5); var, own = [], []
for j, c in enumerate(plain):
    for v, m in variants(c, V, rng, target=tgt): var.append((fin(v), m)); own.append(j)
nb.backend.set_options(noise_model=nm0, seed_simulator=11)
res = nb.backend.run([v for v, _ in var], shots=SH // V).result()
C = [dict() for _ in plain]
for k, (v, m) in enumerate(var):
    for key, n in unflip(res.get_counts(k), m).items(): C[own[k]][key] = C[own[k]].get(key, 0) + n
print(f'{name}: {n2} 2q errors twirled, {nro} readout symmetrised; N={N}, V={V}, shots {SH}')
for j in range(N):
    m = [B.readout_stats(x[j], infos[j], meta['meas'], meta['discard'])[0] for x in (A, Bt, C)]
    print(f' circ {j}: hel exact a {B.hellinger(A[j], exact[j]):.4f} b {B.hellinger(Bt[j], exact[j]):.4f} c {B.hellinger(C[j], exact[j]):.4f} | '
          f'hel(b,c) {B.hellinger(Bt[j], C[j]):.5f} hel(a,b) {B.hellinger(A[j], Bt[j]):.5f} | signal m a {m[0]:+.4f} b {m[1]:+.4f} c {m[2]:+.4f} exact {B.readout_stats(exact[j], infos[j], meta["meas"], meta["discard"])[0]:+.4f}', flush=True)

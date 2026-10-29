# Debiasing in the limit of infinitely many variants, for a results dir written by backends.py (qlimit map).
# Averaging over random Pauli twirls of a Clifford 2q gate (cz/ecr/cx) turns its error channel E into the Pauli channel
# p_Q = sum_k |tr(Q K_k) / d|^2 (K_k: Kraus operators of E); averaging over random readout flips turns each qubit's
# assignment matrix into a symmetric one with e = (p(1|0) + p(0|1)) / 2. So the noise model is rewritten once and every
# circuit runs once (5 x shots, split at random into 5 reps: the density matrix does not depend on shots).
# 1q gate errors and the honest profile's idle relaxation channels are left as they are (as in debias.py).
#   PYTHONPATH=.:analysis/bench python analysis/bench/twirl_limit.py results/bench_b2/<set> --backends ibm_miami
import argparse, copy, itertools, json, pickle, time
from pathlib import Path
import numpy as np
from qiskit import qpy
from qiskit.quantum_info import Kraus, Pauli
from qiskit_aer.noise import pauli_error, ReadoutError
import qiskit_ibm_runtime.fake_provider as fake_provider

from modules.utils.general import setup_exp, resolve_eval_config
from modules.compilation.quantum.emul import BackendManager
import bench_lib as B
from backends import NOISY, gpu_energy_mj


def pauli_twirl(err):
    kr = Kraus(err.to_quantumchannel()).data
    n = err.num_qubits; d = 2 ** n
    labels = [''.join(p) for p in itertools.product('IXYZ', repeat=n)]
    probs = [sum(abs(np.trace(Pauli(l).to_matrix().conj().T @ k) / d) ** 2 for k in kr) for l in labels]
    probs = np.clip(np.real(probs), 0, None); probs /= probs.sum()
    return pauli_error([(l, p) for l, p in zip(labels, probs) if p > 1e-15])


def twirled_noise_model(nm):
    nm = copy.deepcopy(nm)
    n2 = 0
    for gate, errs in nm._local_quantum_errors.items():
        for qubits, err in list(errs.items()):
            if len(qubits) == 2:
                errs[qubits] = pauli_twirl(err); n2 += 1
    for qubits, ro in list(nm._local_readout_errors.items()):
        p = np.asarray(ro.probabilities)
        e = (p[0, 1] + p[1, 0]) / 2
        nm._local_readout_errors[qubits] = ReadoutError([[1 - e, e], [e, 1 - e]])
    return nm, n2, len(nm._local_readout_errors)


def split_counts(counts, parts, rng):
    keys = list(counts); shots = np.repeat(np.arange(len(keys)), [counts[k] for k in keys]); rng.shuffle(shots)
    return [{keys[i]: int(c) for i, c in enumerate(np.bincount(chunk, minlength=len(keys))) if c} for chunk in np.array_split(shots, parts)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('--backends', default='ibm_miami,honest_miami,ibm_kingston')
    p.add_argument('--threads', type=int, default=0)
    a = p.parse_args()
    d = Path(a.dir)
    meta = json.load(open(d / 'summary.json'))['meta']
    assert meta.get('full_map') is False, 'twirl_limit expects a qlimit-map run (bench_b2)'
    raw = pickle.load(open(d / 'raw.pkl', 'rb'))
    circs = qpy.load(open(d / 'logical.qpy', 'rb'))
    infos, ex, n_pairs = raw['infos'], raw['exact'], meta['pairs']
    config, _, _ = setup_exp(resolve_eval_config(None, Path(meta['checkpoint'])), seed=0)
    config['emulate'].update(set=meta['eval_set'], measurement_method=meta['meas'])
    for name in a.backends.split(','):
        out = d / f'summary_debiasT_{name}.json'
        if out.exists():
            raise FileExistsError(out)
        profile, cls = NOISY[name]
        cfg = copy.deepcopy(config)
        cfg['emulate'].update(noise=True, noise_profile=profile, backend='density_matrix',
                              aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False, 'max_parallel_threads': a.threads})
        nb = BackendManager(cfg, meta['qlimit'], hardware_profile=getattr(fake_provider, cls)())
        nm, n2, nro = twirled_noise_model(nb.backend.options.noise_model)
        nb.backend.set_options(noise_model=nm, seed_simulator=4242)
        idle = nb.honest['idle']
        nb.honest['idle'] = False
        plain = nb.transpile(circs, batch_size=256, optimization_level=1)   # same seed -> same layout as backends.py
        nb.honest['idle'] = idle
        tc = nb.transpile(circs, batch_size=256, optimization_level=1) if (profile == 'honest' and idle) else plain
        e0, t0 = gpu_energy_mj(), time.time()
        allc = []
        for i in range(0, len(tc), 256):
            res = nb.backend.run(tc[i:i + 256], shots=meta['shots'] * meta['reps']).result()
            allc += [dict(res.get_counts(j)) for j in range(len(tc[i:i + 256]))]
        rng = np.random.default_rng(99)
        parts = [split_counts(c, meta['reps'], rng) for c in allc]
        rep_counts = [[parts[j][r] for j in range(len(allc))] for r in range(meta['reps'])]
        row, extra = B.backend_metrics(f'{name}_debias_twirl', rep_counts, infos, ex, meta['meas'], meta['discard'], n_pairs, meta['shots'])
        row.update(stats=B.summarise_stats(B.circ_stats(plain, target=nb.hardware_profile.target)), sim_wall_s=time.time() - t0,
                   sim_gpu_energy_J=None, profile=nb.hardware_profile.name, noise_profile=profile,
                   mitigation=f'debias, twirl limit: {n2} 2q errors Pauli-twirled, {nro} readout errors symmetrised')
        print(f'{name}_debias_twirl acc {row["acc_mean"]} +- {row["acc_std"]} hellinger {row["hellinger_fid_mean"]} ({time.time() - t0:.0f}s, {n2} 2q errors twirled)', flush=True)
        json.dump({'meta': meta, 'rows': [row]}, open(out, 'w'), indent=1, default=str)
        pickle.dump({f'{name}_debias_twirl': {'extra': extra, 'counts_rep0': rep_counts[0]}}, open(d / f'raw_debiasT_{name}.pkl', 'wb'))
    print('done', d, flush=True)


if __name__ == '__main__':
    main()

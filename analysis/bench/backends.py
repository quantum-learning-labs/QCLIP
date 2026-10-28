# Backend benchmark of one checkpoint and one readout (destructive swap test or compute-uncompute) on the
# SVO-probes swap set: exact simulation (infinite shots + finite-shot sampling), Aer with IBM fake-backend
# noise (FakeMiami as in emulate.py, and FakeKingston) and the honest profile. Writes the logical circuits (QPY)
# for analysis/bench/ionq_run.py. Usage (from the repo root, GPU node):
#   PYTHONPATH=.:analysis/bench python analysis/bench/backends.py -cp <best.pt> --meas destructive_swap -o results/bench_b1/<name>
import argparse, copy, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from qiskit import qpy
import qiskit_ibm_runtime.fake_provider as fake_provider

from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager
import bench_lib as B

NOISY = {'ibm_miami': ('backend', 'FakeMiami'), 'honest_miami': ('honest', 'FakeMiami'), 'ibm_kingston': ('backend', 'FakeKingston')}
LOGICAL_BASIS = ['rz', 'sx', 'x', 'cx']
IONQ_BASIS = ['rz', 'ry', 'rx', 'rxx']   # all-to-all trapped ion: MS = rxx(pi/2) up to 1q gates


def gpu_energy_mj():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-cp', '--checkpoint', required=True)
    p.add_argument('--meas', choices=['destructive_swap', 'compute_uncompute'], required=True)
    p.add_argument('--shots', type=int, default=8192, help='shots per circuit per rep')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--backends', default='exact,ibm_miami,honest_miami,ibm_kingston')
    p.add_argument('--method', default='density_matrix', help='Aer method for the noisy backends')
    p.add_argument('--eval_set', default='swap')
    p.add_argument('-l', '--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    ckpt = Path(a.checkpoint)
    config, DEV, SEED = setup_exp(resolve_eval_config(None, ckpt), seed=a.seed)
    config['emulate'].update(set=a.eval_set, measurement_method=a.meas, noise=False)
    discard = config['text'].get('discard', False)

    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    c_train, c_val, c_eval = de.compile_text('train'), de.compile_text('val'), de.compile_text(a.eval_set)
    einsum_data = de.describe_einsum(text_model, c_eval, return_metrics=True)
    qlimit = einsum_data['max'][0] + (config['embedding_qubits'] if a.meas == 'destructive_swap' else 0)
    de.text_init(text_model, pd.concat([c_train, c_val], ignore_index=True))
    de.image_init(image_model)
    ck = CheckpointManager.load_model(ckpt, image_model, text_model, DEV)
    image_model.eval(); text_model.eval()

    # logical circuits: the emulator's own assembly, transpilation switched off
    bm = BackendManager(config, qlimit)
    bm.transpile = lambda circs, **kw: circs
    pos, neg = bm.compile_circuits(de.get_dataset(c_eval, a.eval_set), text_model, image_model, a.limit)
    circs, n_pairs = pos + neg, len(pos)
    infos = [{'out_qs': list(qc.metadata['output_qubits']), 'num_clbits': qc.num_clbits} for qc in circs]
    with open(out / 'logical.qpy', 'wb') as f:
        qpy.dump(circs, f)

    meta = {'checkpoint': str(ckpt), 'epoch': ck.get('epoch'), 'discard': discard, 'meas': a.meas, 'eval_set': a.eval_set,
            'pairs': n_pairs, 'shots': a.shots, 'reps': a.reps, 'method': a.method, 'git': git_state(), 'qlimit': qlimit}
    print(f'{n_pairs} pairs, {len(circs)} circuits, discard={discard}, meas={a.meas}', flush=True)

    # exact distributions (CPU statevector of the logical circuit)
    t = time.time()
    probs = [B.exact_probs(qc) for qc in circs]
    st = [B.readout_stats(pr, i, a.meas, discard) for pr, i in zip(probs, infos)]
    ex = {'probs': probs, 'm': [x[0] for x in st], 'a': [x[1] for x in st], 'score': [x[2] for x in st]}
    sc = np.array(ex['score'])
    exact_row = {'backend': 'exact_inf', 'acc_mean': float(np.mean(sc[:n_pairs] > sc[n_pairs:])), 'wall_s': time.time() - t,
                 'acceptance_mean': float(np.mean(ex['a'])), 'acceptance_min': float(np.min(ex['a'])),
                 'margin_median': float(np.median(np.abs(np.array(ex['m'][:n_pairs]) - np.array(ex['m'][n_pairs:])))),
                 'output_register_dist': B.mean_dists([B.register_dists(pr, i, a.meas, discard)[0] for pr, i in zip(probs, infos)]),
                 'aux_hamming_dist': {} if discard else B.mean_dists([B.register_dists(pr, i, a.meas, discard)[1] for pr, i in zip(probs, infos)])}
    cur = B.shot_curve(ex['m'][:n_pairs], ex['a'][:n_pairs], ex['m'][n_pairs:], ex['a'][n_pairs:], a.meas)
    exact_row.update(shots_per_circuit_95=cur['shots_per_circuit_95'], shots_per_circuit_99=cur['shots_per_circuit_99'],
                     acc_at_shots={s: cur['curve'][s] for s in (64, 256, 1024, 4096, 16384, 65536)})
    exact_row['stats_logical'] = B.summarise_stats(B.circ_stats(circs, basis=LOGICAL_BASIS))
    exact_row['stats_ionq_native'] = B.summarise_stats(B.circ_stats(circs, basis=IONQ_BASIS))
    print('exact accuracy', exact_row['acc_mean'], flush=True)
    rows, raw = [exact_row], {'exact': ex, 'exact_curve': cur['curve'], 'infos': infos}

    backends = a.backends.split(',')
    if 'exact' in backends:   # noiseless finite-shot sampling from the exact distributions
        rng = np.random.default_rng(a.seed)
        rep_counts = []
        for r in range(a.reps):
            reps = []
            for pr in probs:
                keys = list(pr); pv = np.array([pr[k] for k in keys]); pv /= pv.sum()
                reps.append({k: int(c) for k, c in zip(keys, rng.multinomial(a.shots, pv)) if c})
            rep_counts.append(reps)
        row, extra = B.backend_metrics('exact_shots', rep_counts, infos, ex, a.meas, discard, n_pairs, a.shots)
        rows.append(row); raw['exact_shots'] = {'extra': extra, 'counts_rep0': rep_counts[0]}

    for name in [b for b in backends if b in NOISY]:
        profile, cls = NOISY[name]
        cfg = copy.deepcopy(config)
        cfg['emulate'].update(noise=True, noise_profile=profile, backend=a.method,
                              aer_options={'batched_shots_gpu': False, 'shot_branching_enable': False})
        nb = BackendManager(cfg, None, hardware_profile=getattr(fake_provider, cls)())
        idle = nb.honest['idle']
        nb.honest['idle'] = False   # gate-level circuit for the statistics (same seed -> same layout and routing)
        plain = nb.transpile(circs, batch_size=256, optimization_level=1)
        nb.honest['idle'] = idle
        tc = nb.transpile(circs, batch_size=256, optimization_level=1) if (profile == 'honest' and idle) else plain
        stats = B.circ_stats(plain, target=nb.hardware_profile.target)
        e0, t0 = gpu_energy_mj(), time.time()
        rep_counts = []
        for r in range(a.reps):
            nb.backend.set_options(seed_simulator=1000 * (r + 1) + a.seed)
            cnt = []
            for i in range(0, len(tc), 64):
                res = nb.backend.run(tc[i:i + 64], shots=a.shots).result()
                cnt += [dict(res.get_counts(j)) for j in range(len(tc[i:i + 64]))]
            rep_counts.append(cnt)
            print(f'{name} rep {r}: {time.time() - t0:.0f}s', flush=True)
        e1, wall = gpu_energy_mj(), time.time() - t0
        row, extra = B.backend_metrics(name, rep_counts, infos, ex, a.meas, discard, n_pairs, a.shots)
        row.update(stats=B.summarise_stats(stats), sim_wall_s=wall, sim_gpu_energy_J=(e1 - e0) / 1e3 if e0 is not None and e1 is not None else None,
                   profile=nb.hardware_profile.name, noise_profile=profile)
        print(name, 'acc', row['acc_mean'], '+-', row['acc_std'], 'hellinger', row['hellinger_fid_mean'], flush=True)
        rows.append(row); raw[name] = {'extra': extra, 'counts_rep0': rep_counts[0], 'stats': stats}
        json.dump({'meta': meta, 'rows': rows}, open(out / 'summary.json', 'w'), indent=1, default=str)

    json.dump({'meta': meta, 'rows': rows}, open(out / 'summary.json', 'w'), indent=1, default=str)
    pickle.dump(raw, open(out / 'raw.pkl', 'wb'))
    print('done', out, flush=True)


if __name__ == '__main__':
    main()

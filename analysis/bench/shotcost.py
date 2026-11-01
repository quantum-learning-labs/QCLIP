# Shot-cost study, step 1 (exact, no sampling): per checkpoint and eval set, the compute-uncompute logical circuits of
# every pair and their exact readout statistics: signal m (= P(overlap | accepted), identical for the swap test, checked on
# bench_b2), acceptance a (1 for discard), k = number of post-selected / discarded text qubits, circuit size.
# The swap-test and CU shot curves follow analytically from (m, a) (shotcost_report.py). Several checkpoints per call.
# Usage (repo root, pool node):
#   PYTHONPATH=.:analysis/bench python analysis/bench/shotcost.py -o results/shotcost/e0 --sets std,swap_full -cp a/best.pt b/best.pt
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd

from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config, git_state
from modules.data_pipeline.engine import DataEngine
from modules.compilation.quantum.emul import BackendManager
import bench_lib as B

SWAP_SETS = {'swap_full': 'data/svo-probes/processed/swap_full.pkl', 'swap_animate': 'data/svo-probes/processed/swap_animate.pkl'}


def run_one(ckpt, sets, out, limit, seed):
    name = ckpt.parent.name
    config, DEV, _ = setup_exp(resolve_eval_config(None, ckpt), seed=seed)
    for s, path in SWAP_SETS.items():   # extra swap sets: same class/task as 'swap', other caption file
        if s in sets and s not in config['dataset']['test']:
            config['dataset']['test'][s] = dict(config['dataset']['test']['swap'], text_path=path)
    discard = config['text'].get('discard', False)
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    c_train, c_val = de.compile_text('train'), de.compile_text('val')
    de.text_init(text_model, pd.concat([c_train, c_val], ignore_index=True))
    de.image_init(image_model)
    ck = CheckpointManager.load_model(ckpt, image_model, text_model, DEV)
    image_model.eval(); text_model.eval()
    for s in sets:
        f = out / f'{name}__{s}.npz'
        if f.exists():
            print('exists, skip', f, flush=True); continue
        t = time.time()
        config['emulate'].update(set=s, measurement_method='compute_uncompute', noise=False)
        c_eval = de.compile_text(s)
        qlimit = de.describe_einsum(text_model, c_eval, return_metrics=True)['max'][0]   # CU: text register only (as backends.py)
        bm = BackendManager(config, qlimit)
        bm.transpile = lambda circs, **kw: circs
        pos, neg = bm.compile_circuits(de.get_dataset(c_eval, s), text_model, image_model, limit)
        circs, n = pos + neg, len(pos)
        if limit is None and n != len(c_eval) and s == 'std':
            print(f'WARNING {name} {s}: {n} circuit pairs from {len(c_eval)} rows', flush=True)
        m, a, k, nq = [], [], [], []
        for qc in circs:
            info = {'out_qs': list(qc.metadata['output_qubits']), 'num_clbits': qc.num_clbits}
            mi, ai, _ = B.readout_stats(B.exact_probs(qc), info, 'compute_uncompute', discard)
            m.append(mi); a.append(ai); nq.append(qc.num_qubits)
            # PS: aux = measured text qubits outside the output; discard: unmeasured qubits (traced out)
            k.append(qc.num_clbits - len(info['out_qs']) if not discard else qc.num_qubits - qc.num_clbits)
        m, a, k, nq = map(np.asarray, (m, a, k, nq))
        np.savez(f, m_pos=m[:n], m_neg=m[n:], a_pos=a[:n], a_neg=a[n:], k_pos=k[:n], k_neg=k[n:], nq_pos=nq[:n], nq_neg=nq[n:])
        meta = {'checkpoint': str(ckpt), 'epoch': ck.get('epoch'), 'discard': discard, 'set': s, 'pairs': n,
                'text_n': config['text'].get('n'), 'embedding_qubits': config['embedding_qubits'], 'git': git_state(),
                'exact_acc': float(np.mean((m[:n] > m[n:]) + 0.5 * (m[:n] == m[n:]))), 'accept_mean': float(a.mean()),
                'k_mean': float(k.mean()), 'nq_max': int(nq.max()), 'wall_s': time.time() - t}
        json.dump(meta, open(f.with_suffix('.json'), 'w'), indent=1, default=str)
        print(name, s, {x: meta[x] for x in ('pairs', 'exact_acc', 'accept_mean', 'k_mean', 'nq_max', 'wall_s')}, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    p.add_argument('--sets', default='std,swap_full')
    p.add_argument('-l', '--limit', type=int, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--out', required=True)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for c in a.checkpoints:
        run_one(Path(c), a.sets.split(','), out, a.limit, a.seed)
    print('done', flush=True)


if __name__ == '__main__':
    main()

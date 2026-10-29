# IonQ cloud simulator with a device noise model (free; NEVER a QPU backend) on the logical circuits written by
# backends.py. Adds an 'ionq_<noise>' row to <dir>/summary.json (as summary_ionq.json; summary.json is not rewritten).
#   PYTHONPATH=.:analysis/bench python analysis/bench/ionq_run.py <dir> --noise forte-1 --reps 5 --shots 8192
import argparse, json, os, pickle, time
from pathlib import Path
import numpy as np
from qiskit import qpy, transpile
from qiskit_ionq import IonQProvider
import bench_lib as B

IONQ_BASIS = ['rz', 'ry', 'rx', 'rxx']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dir')
    p.add_argument('--noise', default='forte-1')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--shots', type=int, default=8192)
    p.add_argument('--batch', type=int, default=40, help='circuits per IonQ job')
    p.add_argument('--ideal_check', type=int, default=4, help='circuits run noiseless first, compared to the exact distribution')
    p.add_argument('--debias', type=int, default=0, help='IonQ-style debiasing done here (the simulator ignores error_mitigation): '
                   'this many random variants per circuit per rep (2q Pauli twirl + readout flips), shots / variants each')
    a = p.parse_args()
    d = Path(a.dir)
    tag = a.noise + ('_debias' if a.debias else '')
    out = d / f'summary_ionq_{tag}.json'
    if out.exists():
        raise FileExistsError(out)
    summ = json.load(open(d / 'summary.json')); meta = summ['meta']
    raw = pickle.load(open(d / 'raw.pkl', 'rb'))
    circs = qpy.load(open(d / 'logical.qpy', 'rb'))
    infos, ex, n_pairs = raw['infos'], raw['exact'], meta['pairs']

    key = open(os.path.expanduser('~/.ionq_key')).read().strip()
    sim = IonQProvider(token=key).get_backend('simulator')   # simulator only: no QPU anywhere in this script
    assert 'qpu' not in sim.name.lower()
    tc = transpile(circs, backend=sim, optimization_level=1, seed_transpiler=0)

    def run(cs, noise, shots, label):
        # submit every batch first (the simulator queue, not the simulation, dominates), then collect in order
        kw = {'noise_model': noise} if noise else {}
        jobs = [sim.run(cs[i:i + a.batch], shots=shots, **kw) for i in range(0, len(cs), a.batch)]
        with open(d / f'ionq_jobs_{tag}.txt', 'a') as f:
            f.write(label + ' ' + ' '.join(j.job_id() for j in jobs) + '\n')
        counts, t = [], time.time()
        for k, job in enumerate(jobs):
            cnt = job.result().get_counts()
            counts += [dict(c) for c in (cnt if isinstance(cnt, list) else [cnt])]
            print(f'{label}: job {k + 1}/{len(jobs)} done after {time.time() - t:.0f}s', flush=True)
        return counts

    # sanity check of the bit order / partial measurement handling: ideal simulator vs exact
    chk = run(tc[:a.ideal_check], None, a.shots, 'ideal_check')
    hel = [B.hellinger(c, ex['probs'][j]) for j, c in enumerate(chk)]
    print('ideal check hellinger vs exact', [round(h, 4) for h in hel], flush=True)
    if min(hel) < 0.97:
        raise SystemExit('ideal IonQ simulator disagrees with the exact distribution: bit order / measurement mapping is wrong')

    t0 = time.time()
    if a.debias:   # every rep: debias variants of every circuit, counts un-flipped and summed per circuit
        from debias import variants, unflip
        rng = np.random.default_rng(12345)
        var, owner = [], []
        for r in range(a.reps):
            for j, qc in enumerate(tc):
                for v, m in variants(qc, a.debias, rng):
                    var.append((v, m)); owner.append((r, j))
        allc = run([v for v, _ in var], a.noise, a.shots // a.debias, f'{a.reps} reps x {a.debias} variants')
        rep_counts = [[dict() for _ in tc] for _ in range(a.reps)]
        for (r, j), (v, m), c in zip(owner, var, allc):
            for k, n in unflip(c, m).items():
                rep_counts[r][j][k] = rep_counts[r][j].get(k, 0) + n
    else:
        allc = run(tc * a.reps, a.noise, a.shots, f'{a.reps} reps')   # one submission round for all reps
        rep_counts = [allc[r * len(tc):(r + 1) * len(tc)] for r in range(a.reps)]
    row, extra = B.backend_metrics(f'ionq_{tag}', rep_counts, infos, ex, meta['meas'], meta['discard'], n_pairs, a.shots)
    row.update(wall_s=time.time() - t0, ideal_check_hellinger=hel, noise_model=a.noise,
               mitigation=f'debias: 2q Pauli twirl + readout flips, {a.debias} variants' if a.debias else None)
    json.dump({'meta': meta, 'rows': [row]}, open(out, 'w'), indent=1, default=str)
    pickle.dump({'extra': extra, 'counts': rep_counts}, open(d / f'raw_ionq_{tag}.pkl', 'wb'))
    print('done', row['acc_mean'], row['acc_std'], row['hellinger_fid_mean'], flush=True)


if __name__ == '__main__':
    main()

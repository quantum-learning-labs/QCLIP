# Shared pieces of the backend benchmark (analysis/bench): scoring of counts, exact distributions, circuit
# statistics, and the per-backend metrics. Scores follow modules/compilation/quantum/emul.py exactly
# (Emulator._eval_compute_uncompute / _eval_destructive_swap), so accuracies match emulate.py.
import math
import numpy as np
from scipy.stats import norm, spearmanr
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

TWO_Q = {'cx', 'cz', 'ecr', 'rzz', 'rxx', 'ryy', 'swap', 'iswap', 'ms', 'zz', 'cp', 'crx', 'cry', 'crz', 'cu', 'cswap'}


def meas_map(qc):
    # clbit index -> qubit index of the final measurements
    m = {}
    for ins in qc.data:
        if ins.operation.name == 'measure':
            m[qc.find_bit(ins.clbits[0]).index] = qc.find_bit(ins.qubits[0]).index
    return [m[c] for c in range(qc.num_clbits)]


def exact_probs(qc):
    # exact distribution over the measured clbits, keyed like Qiskit counts (clbit 0 rightmost)
    qargs = meas_map(qc)
    sv = Statevector.from_instruction(qc.remove_final_measurements(inplace=False))
    return {k: float(v) for k, v in sv.probabilities_dict(qargs=qargs).items() if v > 1e-15}


def readout_stats(counts, info, meas, discard):
    """Per circuit: (signal mean among accepted shots m, acceptance a, emulate.py score).
    swap: m = parity mean (+1/-1), score = max(0, m); compute_uncompute: m = P(all zero | accepted)."""
    out_qs, ncl = info['out_qs'], info['num_clbits']
    n_out = len(out_qs)
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0, 0.0, 0.0
    acc = sig = 0.0
    if meas == 'destructive_swap':
        if discard:
            for k, c in counts.items():
                b = k[::-1]
                par = sum(int(b[i]) * int(b[n_out + i]) for i in range(n_out)) % 2
                sig += c * (1 - 2 * par); acc += c
        else:
            n_txt = ncl - n_out
            aux = [q for q in range(n_txt) if q not in out_qs]
            for k, c in counts.items():
                b = k[::-1]
                if all(b[q] == '0' for q in aux):
                    par = sum(int(b[out_qs[i]]) * int(b[n_txt + i]) for i in range(n_out)) % 2
                    sig += c * (1 - 2 * par); acc += c
        m = sig / acc if acc else 0.0
        return m, acc / tot, max(0.0, m) if acc else 0.0
    # compute_uncompute
    if discard:
        z = counts.get('0' * ncl, 0.0)
        return z / tot, 1.0, z / tot
    oq = set(out_qs)
    z = counts.get('0' * ncl, 0.0)
    for k, c in counts.items():
        b = k[::-1]
        if all(b[q] == '0' for q in range(ncl) if q not in oq):
            acc += c
    m = z / acc if acc else 0.0
    return m, acc / tot, m


def shot_var(m, meas):
    # per accepted shot variance of the signal estimator
    return max(1 - m * m, 1e-12) if meas == 'destructive_swap' else max(m * (1 - m), 1e-12)


def expected_acc(mp, ap, mn, an, meas, shots):
    # normal approximation of P(score_pos > score_neg) at `shots` per circuit, averaged over pairs
    mp, ap, mn, an = map(np.asarray, (mp, ap, mn, an))
    vp = np.array([shot_var(x, meas) for x in mp]) / np.maximum(shots * ap, 1e-12)
    vn = np.array([shot_var(x, meas) for x in mn]) / np.maximum(shots * an, 1e-12)
    return float(np.mean(norm.cdf((mp - mn) / np.sqrt(vp + vn))))


def shot_curve(mp, ap, mn, an, meas, grid=None):
    grid = grid or [2 ** k for k in range(4, 25)]
    curve = {s: expected_acc(mp, ap, mn, an, meas, s) for s in grid}
    asym = float(np.mean((np.asarray(mp) > np.asarray(mn)) + 0.5 * (np.asarray(mp) == np.asarray(mn))))
    def first(frac):
        tgt = 0.5 + frac * (asym - 0.5)
        return next((s for s in grid if curve[s] >= tgt), None)
    return {'curve': curve, 'asymptotic_acc': asym, 'shots_per_circuit_95': first(0.95), 'shots_per_circuit_99': first(0.99)}


def hellinger(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    bc = sum(math.sqrt(p.get(k, 0) / sp * q.get(k, 0) / sq) for k in keys)
    return bc ** 2


def tvd(p, q):
    keys = set(p) | set(q)
    sp, sq = sum(p.values()) or 1, sum(q.values()) or 1
    return 0.5 * sum(abs(p.get(k, 0) / sp - q.get(k, 0) / sq) for k in keys)


def register_dists(counts, info, meas, discard):
    """Output-register distribution (swap: text-out bits then image bits; CU: output bits) and,
    for post-selection, the Hamming weight of the post-selected (aux) register."""
    out_qs, ncl = info['out_qs'], info['num_clbits']
    n_out = len(out_qs)
    tot = sum(counts.values()) or 1
    reg, hw = {}, {}
    for k, c in counts.items():
        b = k[::-1]
        if meas == 'destructive_swap':
            if discard:
                key = ''.join(b[i] for i in range(n_out)) + '|' + ''.join(b[n_out + i] for i in range(n_out))
                aux = []
            else:
                n_txt = ncl - n_out
                key = ''.join(b[q] for q in out_qs) + '|' + ''.join(b[n_txt + i] for i in range(n_out))
                aux = [b[q] for q in range(n_txt) if q not in out_qs]
        else:
            key = ''.join(b) if discard else ''.join(b[q] for q in out_qs)
            aux = [] if discard else [b[q] for q in range(ncl) if q not in set(out_qs)]
        reg[key] = reg.get(key, 0) + c / tot
        if aux:
            w = sum(x == '1' for x in aux)
            hw[w] = hw.get(w, 0) + c / tot
    return reg, hw


def mean_dists(list_of_dicts):
    acc = {}
    for d in list_of_dicts:
        for k, v in d.items():
            acc[k] = acc.get(k, 0) + v / len(list_of_dicts)
    return dict(sorted(acc.items()))


def circ_stats(circs, target=None, basis=None, coupling=None, seed=0, durations=None):
    """depth / 2q count / active qubits; with a Qiskit target also duration (s) and estimated success
    probability (product of 1 - error over every gate and measurement)."""
    out = []
    for qc in circs:
        if basis is not None:
            qc = transpile(qc, basis_gates=basis, coupling_map=coupling, optimization_level=1, seed_transpiler=seed)
        ops = qc.count_ops()
        n2 = sum(v for k, v in ops.items() if k in TWO_Q)
        n1 = sum(v for k, v in ops.items() if k not in TWO_Q | {'measure', 'barrier', 'delay', 'reset'}
                 and not k.startswith('quantum_channel') and not k.startswith('kraus'))
        act = len({q for ins in qc.data if ins.operation.name not in ('barrier', 'delay') for q in ins.qubits})
        d = {'depth': qc.depth(lambda ins: ins.operation.name not in ('barrier', 'delay')),
             'depth_2q': qc.depth(lambda ins: ins.operation.name in TWO_Q), 'n2q': n2, 'n1q': n1,
             'nmeas': ops.get('measure', 0), 'active_qubits': act}
        if target is not None:
            lesp = 0.0
            for ins in qc.data:
                name = ins.operation.name
                if name in ('barrier', 'delay') or name not in target.operation_names:
                    continue
                qa = tuple(qc.find_bit(q).index for q in ins.qubits)
                p = target[name].get(qa)
                if p is not None and p.error:
                    lesp += math.log(max(1 - p.error, 1e-12))
            d['esp'] = math.exp(lesp)
            try:
                d['duration_s'] = float(qc.estimate_duration(target, unit='s'))
            except Exception:
                d['duration_s'] = None
        if durations is not None:   # IonQ-style model: dict with t1q, t2q, tmeas (s) and error rates e1q, e2q, espam
            d['duration_s'] = durations['t1q'] * d['depth'] + durations['t2q'] * d['depth_2q'] + durations['tmeas']
            d['esp'] = (1 - durations['e1q']) ** n1 * (1 - durations['e2q']) ** n2 * (1 - durations['espam']) ** d['nmeas']
        out.append(d)
    return out


def summarise_stats(stats):
    keys = stats[0].keys()
    s = {}
    for k in keys:
        v = [x[k] for x in stats if x[k] is not None]
        if v:
            s[k + '_mean'] = float(np.mean(v)); s[k + '_max'] = float(np.max(v))
    return s


def backend_metrics(name, rep_counts, infos, exact, meas, discard, n_pairs, shots):
    """rep_counts: list over reps of list over circuits (pos circuits then neg) of counts dicts.
    exact: dict with per-circuit exact probs and m/a. Returns the summary row and per-pair data."""
    reps = []
    for counts in rep_counts:
        st = [readout_stats(c, i, meas, discard) for c, i in zip(counts, infos)]
        sc = np.array([x[2] for x in st])
        reps.append({'acc': float(np.mean(sc[:n_pairs] > sc[n_pairs:])), 'scores': sc, 'accept': np.array([x[1] for x in st])})
    accs = np.array([r['acc'] for r in reps])
    # pooled counts over reps = the backend's large-sample distribution
    pooled = []
    for j in range(len(infos)):
        d = {}
        for counts in rep_counts:
            for k, c in counts[j].items():
                d[k] = d.get(k, 0) + c
        pooled.append(d)
    pst = [readout_stats(c, i, meas, discard) for c, i in zip(pooled, infos)]
    m = np.array([x[0] for x in pst]); a = np.array([x[1] for x in pst])
    ex_m = np.array(exact['m']); ex_dec = ex_m[:n_pairs] > ex_m[n_pairs:]
    dec_rep = [r['scores'][:n_pairs] > r['scores'][n_pairs:] for r in reps]
    hel = [hellinger(pooled[j], exact['probs'][j]) for j in range(len(infos))]
    tv = [tvd(pooled[j], exact['probs'][j]) for j in range(len(infos))]
    margin = m[:n_pairs] - m[n_pairs:]
    ex_margin = ex_m[:n_pairs] - ex_m[n_pairs:]
    curve = shot_curve(m[:n_pairs], a[:n_pairs], m[n_pairs:], a[n_pairs:], meas)
    regs = [register_dists(c, i, meas, discard) for c, i in zip(pooled, infos)]
    row = {
        'backend': name, 'reps': len(reps), 'acc_mean': float(accs.mean()), 'acc_std': float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        'acc_var': float(accs.var(ddof=1)) if len(accs) > 1 else 0.0, 'acc_reps': accs.tolist(),
        'acc_se_pairs': float(math.sqrt(max(accs.mean() * (1 - accs.mean()), 1e-12) / n_pairs)),
        'agree_with_exact': float(np.mean([np.mean(d == ex_dec) for d in dec_rep])),
        'score_spearman_vs_exact': float(spearmanr(m, ex_m).correlation),
        'margin_ratio_vs_exact': float(np.median(margin[np.abs(ex_margin) > 1e-6] / ex_margin[np.abs(ex_margin) > 1e-6])),
        'margin_median': float(np.median(np.abs(margin))), 'hellinger_fid_mean': float(np.mean(hel)), 'tvd_mean': float(np.mean(tv)),
        'acceptance_mean': float(a.mean()), 'acceptance_min': float(a.min()),
        'asymptotic_acc': curve['asymptotic_acc'], 'shots_per_circuit_95': curve['shots_per_circuit_95'],
        'shots_per_circuit_99': curve['shots_per_circuit_99'],
        'acc_at_shots': {s: curve['curve'][s] for s in (64, 256, 1024, 4096, 16384, 65536)},
        # pairs whose margin is under 2 sigma of the estimator at the shots actually used (decided by shot noise)
        'frac_pairs_below_shot_noise': float(np.mean(np.abs(margin) < 2 * np.sqrt(
            np.array([shot_var(x, meas) for x in m[:n_pairs]]) / np.maximum(shots * a[:n_pairs], 1e-12)
            + np.array([shot_var(x, meas) for x in m[n_pairs:]]) / np.maximum(shots * a[n_pairs:], 1e-12)))),
        'output_register_dist': mean_dists([r[0] for r in regs]),
        'aux_hamming_dist': mean_dists([r[1] for r in regs]) if not discard else {},
    }
    return row, {'curve': curve['curve'], 'hellinger': hel, 'tvd': tv, 'm': m.tolist(), 'a': a.tolist()}

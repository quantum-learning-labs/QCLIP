# E5 calibration: how much does Kingston noise shrink the decision margin? Per noisy run (e6_sc4*, profile ibm_kingston, 2^20 shots,
# CU readout) against the exact run of the same checkpoint (e4_sc4*): margin d = m_pos - m_neg with m = U0/P0 (post-selection) or sum U
# (discard); F_eff = least-squares slope of d_noisy on d_exact (through 0). Shots to resolve a margin scale ~ 1/F_eff^2 (and 1/acceptance for PS).
# usage: python analysis/bench/calib_noise.py -o results/shotcost/adv/noise_calib.csv
import argparse, csv, glob, json, re
from pathlib import Path
import numpy as np

R = Path('results/shotcost')


def margins(z, discard):
    P, U = z['P'], z['U']
    m = U.sum(-1) if discard else U[:, :, 0] / np.maximum(P[:, :, 0], 1e-300)
    return m[:, 0] - m[:, 1], P[:, :, 0].mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('-o', '--out', required=True); a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(open(out, 'w', newline=''))
    w.writerow(['ckpt', 'e', 'n', 'method', 'seed', 'nq_mean', 'pairs', 'F_eff', 'F_eff_r2', 'acc_exact', 'acc_noisy', 'accept_exact', 'accept_noisy', 'noisy_file'])
    seen = set()
    for f in sorted(glob.glob(str(R / 'e6_sc4_rerun/*__ibm_kingston.npz'))) + sorted(glob.glob(str(R / 'e6_sc4/*__ibm_kingston.npz'))):   # reruns first
        name = Path(f).name.replace('__ibm_kingston.npz', '')
        if name in seen: continue
        discard = '_discard_' in name
        ex_name = name.replace('_ws_', '_')               # warm-start runs are stored without the _ws tag in e4
        cands = [p for d in ('e4_sc4', 'e4_sc4_n3tn', 'e4_sc4_n4tn') for p in (R / d / f'{name}__std.npz', R / d / f'{ex_name}__std.npz') if p.exists()]
        if not cands: print('no exact run for', name); continue
        zn, ze = np.load(f), np.load(cands[0])
        common, i_n, i_e = np.intersect1d(zn['idx'], ze['idx'], return_indices=True)
        if len(common) < 50: print('too few common pairs', name, len(common)); continue
        dn, an = margins({k: zn[k][i_n] for k in ('P', 'U')}, discard)
        de, ae = margins({k: ze[k][i_e] for k in ('P', 'U')}, discard)
        F = float((dn * de).sum() / (de * de).sum()); r2 = float(1 - ((dn - F * de) ** 2).sum() / ((dn - dn.mean()) ** 2).sum())
        e = int(re.search(r'_e(\d+)_', name).group(1)); n = int(re.search(r'_n(\d+)_', name).group(1)); s = int(name[-1])
        method = 'ps' if not discard else ('warm' if '_ws_' in name else 'plain')
        nq = float((zn['k'][i_n] + e).mean())
        w.writerow([name, e, n, method, s, nq, len(common), F, r2, float((de > 0).mean()), float((dn > 0).mean()), ae, an, f])
        seen.add(name); print(f'{name}: nq {nq:.1f} F_eff {F:.3f} (r2 {r2:.2f}) acc {(de > 0).mean():.3f} -> {(dn > 0).mean():.3f}', flush=True)


if __name__ == '__main__':
    main()

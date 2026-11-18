# F_eff of noisy calibration runs against the exact runs of the same models (same definition as calib_noise.py):
# margin d = m_pos - m_neg with m = U0/P0 (post-selection); F_eff = least-squares slope of d_noisy on d_exact through 0.
#   python calib_fit.py <noisy_dir> <profile_suffix> <exact_dirs comma-separated> <out.csv>
import sys, glob, csv, re
from pathlib import Path
import numpy as np
nd, suf, exd, out = sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4]
def margins(z):
    m = z['U'][:, :, 0] / np.maximum(z['P'][:, :, 0], 1e-300); return m[:, 0] - m[:, 1]
w = csv.writer(open(out, 'w', newline='')); w.writerow(['ckpt', 'e', 'n', 'seed', 'pairs', 'F_eff', 'F_eff_r2', 'acc_exact', 'acc_noisy', 'file'])
for f in sorted(glob.glob(f'{nd}/*__{suf}.npz')):
    name = Path(f).name.replace(f'__{suf}.npz', '')
    ex = [p for d in exd for p in [Path(d) / f'{name}__std.npz'] if p.exists()]
    if not ex: print('no exact run for', name); continue
    zn, ze = np.load(f), np.load(ex[0])
    common, i_n, i_e = np.intersect1d(zn['idx'], ze['idx'], return_indices=True)
    if len(common) < 30: print('too few common pairs', name, len(common)); continue
    dn, de = margins({k: zn[k][i_n] for k in ('P', 'U')}), margins({k: ze[k][i_e] for k in ('P', 'U')})
    F = float((dn * de).sum() / (de * de).sum()); r2 = float(1 - ((dn - F * de) ** 2).sum() / ((dn - dn.mean()) ** 2).sum())
    e, n, s = int(re.search(r'_e(\d+)_', name).group(1)), int(re.search(r'_n(\d+)_', name).group(1)), int(name[-1])
    w.writerow([name, e, n, s, len(common), F, r2, float((de > 0).mean()), float((dn > 0).mean()), f])
    print(f'{name}: pairs {len(common)} F_eff {F:.3f} r2 {r2:.2f} acc {(de > 0).mean():.3f} -> {(dn > 0).mean():.3f}', flush=True)

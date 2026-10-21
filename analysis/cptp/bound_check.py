# Check the margin bound |<a|rho|a> - <b|rho|b>| <= sqrt(2 (P - 1/d) (1 - |<a|b>|^2)) on per-pair data (v1 gap_analysis pickles).
# usage: bound_check.py <pkl>...
import sys, pickle, numpy as np, pandas as pd
d = 8
for f in sys.argv[1:]:
    R = pickle.load(open(f, 'rb'))
    for disc in (True, False):
        dfs = [r['df'] for r in R if r['split'] == 'val' and r['discard'] == disc]
        if not dfs: continue
        D = pd.concat(dfs)
        bound = np.sqrt(2 * np.clip(D.pur_text - 1 / d, 0, None) * np.clip(1 - D.img_pair_hs, 0, None))
        m = D.hs_margin.abs(); ratio = m / bound.clip(lower=1e-9)
        q = pd.qcut(bound, 4, labels=['Q1 smallest bound', 'Q2', 'Q3', 'Q4 largest'])
        print(f"{'discard' if disc else 'nodiscard'}: pairs {len(D)}, violations {(m > bound + 1e-5).sum()}, "
              f"median |margin|/bound {ratio.median():.3f}, 90th pct {ratio.quantile(.9):.3f}, median bound {bound.median():.3f}")
        print(D.assign(q=q).groupby('q', observed=True).agg(bound=('pur_text', lambda s: float('nan')), acc=('hs_correct', 'mean'),
              median_margin=('hs_margin', 'median')).drop(columns='bound').join(pd.Series(bound).groupby(q, observed=True).median().rename('median_bound')).round(3))

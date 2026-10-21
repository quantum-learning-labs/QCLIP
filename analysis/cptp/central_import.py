# Import every offline analysis / emulation result into the central MLflow DB (served on tails:5056).
# Idempotent: each record carries tags.source_key; records already present are skipped.
# usage: central_import.py <mlflow sqlite db> <staging dir>
#   staging dir layout: gap/{dyn_summary*.csv, readout_swap_*.csv, grad_probe.json, rs_base.pkl.summary.json}
#                       v1/base_{nodiscard,discard}.pkl   emu/*.json (shot sweeps)   emu_noise/*.json (noise runs)
import sys, os, glob, json, csv, pickle, logging, time
import numpy as np, pandas as pd, mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
logging.getLogger('mlflow').setLevel(logging.ERROR)
db, S = sys.argv[1], sys.argv[2]
mlflow.set_tracking_uri(f'sqlite:///{db}'); C = MlflowClient()
NOW = int(time.time() * 1000); added = skipped = 0

def exp(name):
    e = C.get_experiment_by_name(name)
    return e.experiment_id if e else C.create_experiment(name)

def log(experiment, run_name, key, params=None, metrics=None, steps=None, tags=None):
    """metrics: {k: v}; steps: [(k, v, step)]"""
    global added, skipped
    eid = exp(experiment)
    if C.search_runs([eid], filter_string=f"tags.source_key = '{key}'", max_results=1):
        skipped += 1; return
    r = C.create_run(eid, run_name=run_name, tags={'source_key': key, **{k: str(v) for k, v in (tags or {}).items()}})
    ms = [Metric(k, float(v), NOW, 0) for k, v in (metrics or {}).items() if v is not None and np.isfinite(float(v))]
    ms += [Metric(k, float(v), NOW, int(s)) for k, v, s in (steps or []) if v is not None and np.isfinite(float(v))]
    ps = [Param(k, str(v)[:500]) for k, v in (params or {}).items()]
    for i in range(0, max(len(ms), len(ps), 1), 900):
        C.log_batch(r.info.run_id, metrics=ms[i:i + 900], params=ps[i:i + 100] if i == 0 else [])
    if len(ps) > 100:
        for i in range(100, len(ps), 100): C.log_batch(r.info.run_id, params=ps[i:i + 100])
    C.set_terminated(r.info.run_id); added += 1

M = ['acc_ps', 'acc_tr', 'purity', 'p0', 'lmax', 'margin_own', 'score_sd', 'D_global', 'D_trace', 'D_postsel',
     'effdim_tr', 'effdim_ps', 'effdim_img', 'train_acc_own']
ren = {'acc_ps': 'val_acc_postselect_readout', 'acc_tr': 'val_acc_trace_readout', 'purity': 'median_text_purity',
       'p0': 'median_p0', 'lmax': 'median_lambda_max', 'margin_own': 'median_margin_own_readout', 'score_sd': 'score_sd_own_readout',
       'D_global': 'dist_global', 'D_trace': 'dist_trace', 'D_postsel': 'dist_postselect', 'effdim_tr': 'effdim_text_trace',
       'effdim_ps': 'effdim_text_postselect', 'effdim_img': 'effdim_image', 'train_acc_own': 'train_acc_own_readout'}

# 1. readout swap: one run per checkpoint (best.pt)
for f in sorted(glob.glob(f'{S}/gap/readout_swap_*.csv')):
    for row in csv.DictReader(open(f)):
        log('analysis_readout_swap', f"{row['run']}", f"rs:{os.path.basename(f)}:{row['run']}",
            params={'trained_method': row['method'], 'text_layers': row['layers'], 'seed': row['seed'], 'best_epoch': row['epoch'], 'source': os.path.basename(f)},
            metrics={ren[k]: row[k] for k in M if row.get(k) not in (None, '')},
            tags={'analysis': 'E1/E2 readout swap', 'note': 'same trained weights read by post-selection and by partial trace (val set)'})
summ = f'{S}/gap/rs_base.pkl.summary.json'
if os.path.exists(summ):
    for g, d in json.load(open(summ)).items():
        mets = {k + '_mean': v[0] for k, v in d.items() if isinstance(v, list) and len(v) == 2}
        for word, dd in (d.get('dist') or {}).items():
            mets.update({f'{word}_{k}': v for k, v in dd.items()})
        mets.update({f'effdim_{k}': v for k, v in (d.get('eff_dim') or {}).items()})
        log('analysis_readout_swap', f'GROUP {g}', f'rs_group:{g}', params={'group': g, 'n_checkpoints': d.get('n')},
            metrics=mets, tags={'analysis': 'E1/E2 group summary', 'note': 'dist_* = mean distances for val pairs differing in one word'})

# 2. dynamics: one run per training run, step = epoch + 1 (step 0 = initialisation)
for f in sorted(glob.glob(f'{S}/gap/dyn_summary*.csv')):
    df = pd.read_csv(f)
    for run, d in df.groupby('run'):
        steps = [(ren[k], v, int(e) + 1) for _, x in d.iterrows() for k in M if k in x and pd.notna(x[k]) for v, e in [(x[k], x['epoch'])]]
        r0 = d.iloc[0]
        log('analysis_dynamics', run, f'dyn:{run}', params={'trained_method': r0['method'], 'text_layers': r0['layers'], 'seed': r0['seed'], 'source': os.path.basename(f)},
            steps=steps, tags={'analysis': 'E5 training dynamics', 'step_meaning': 'epoch + 1 (0 = initialisation)'})

# 3. gap v1 + margin bound, from the per-pair pickles
d_ = 8
for f in sorted(glob.glob(f'{S}/v1/base_*.pkl')):
    for r in pickle.load(open(f, 'rb')):
        df = r['df']; meth = 'discard' if r['discard'] else 'nodiscard'
        mets = {f'{r["split"]}_acc_{sc}': df[sc + '_correct'].mean() for sc in ('hs', 'fid', 'pauli_cos', 'nhs', 'trdist')}
        if r['split'] == 'val':
            for t in ('subj_neg', 'verb_neg', 'obj_neg'):
                if t in df: mets[f'val_acc_{t}'] = df.hs_correct[df[t]].mean()
            mets.update(val_median_text_purity=df.pur_text.median(), val_median_img_pair_overlap=df.img_pair_hs.median(), val_median_margin=df.hs_margin.median())
        log('analysis_gap_v1', f"{meth}_s{r['seed']}_{r['split']}", f"v1:{meth}:{r['seed']}:{r['split']}",
            params={'trained_method': meth, 'seed': r['seed'], 'split': r['split'], 'checkpoint': r['ckpt'], 'best_epoch': r['epoch']},
            metrics=mets, tags={'analysis': 'v1 gap analysis', 'note': 'accuracy under several similarity functions; per negative type'})
    R = pickle.load(open(f, 'rb'))
    for disc in (True, False):
        D = pd.concat([r['df'] for r in R if r['split'] == 'val' and r['discard'] == disc] or [pd.DataFrame()])
        if D.empty: continue
        bound = np.sqrt(2 * np.clip(D.pur_text - 1 / d_, 0, None) * np.clip(1 - D.img_pair_hs, 0, None)); m = D.hs_margin.abs()
        q = pd.qcut(bound, 4, labels=False); meth = 'discard' if disc else 'nodiscard'
        mets = {'pairs': len(D), 'violations': int((m > bound + 1e-5).sum()), 'median_margin_over_bound': (m / bound.clip(lower=1e-9)).median(),
                'p90_margin_over_bound': (m / bound.clip(lower=1e-9)).quantile(.9), 'median_bound': bound.median()}
        mets.update({f'q{i + 1}_accuracy': D.hs_correct[q == i].mean() for i in range(4)})
        mets.update({f'q{i + 1}_median_bound': bound[q == i].median() for i in range(4)})
        log('analysis_margin_bound', f'{meth} (10 seeds, val)', f'bound:{meth}', params={'trained_method': meth, 'bound': '|Delta| <= sqrt(2 (P - 1/d)(1 - F))', 'd': d_},
            metrics=mets, tags={'analysis': 'E3 margin budget (P1)'})

# 4. gradient probe: one run per (layers, init, readout); step = m (non-sentence wires)
g = f'{S}/gap/grad_probe.json'
if os.path.exists(g):
    for k, d in json.load(open(g)).items():
        L, *init, ro = k.split('_'); init = '_'.join(init)
        name = 'p0' if ro == 'p0' else 'mean_sq_grad'
        steps = [(name, v[0], int(m)) for m, v in d.items()] + [(name + '_std', v[1], int(m)) for m, v in d.items()]
        log('analysis_gradient_probe', k, f'grad:{k}', params={'text_layers': L[1:], 'init': init, 'readout': {'tr': 'partial trace', 'ps': 'post-selection', 'p0': 'p0'}[ro]},
            steps=steps, tags={'analysis': 'E4 gradient probe', 'step_meaning': 'm = number of non-sentence wires'})

# 5./6. emulation
HONEST = "scale_2q=3, scale_1q=2, scale_ro=1.5, scale_t=0.5, zz_rad=0.02, idle=True (HONEST_DEFAULTS in emul.py)"
for exp_name, pattern in (('emulation_shot_sweep', f'{S}/emu/*.json'), ('emulation_noise', f'{S}/emu_noise/*.json')):
    for f in sorted(glob.glob(pattern)):
        name = os.path.basename(f)[:-5]
        if name.startswith('test') or name.startswith('gputest'): continue
        j = json.load(open(f)); res = j['results']; bs = res.get('by_shots') or {str(j.get('shots')): res}
        em = j.get('emulate', {})
        params = {'checkpoint': j.get('checkpoint'), 'trained_method': 'discard' if j.get('discard') else 'nodiscard', 'seed': j.get('seed'),
                  'pairs': j.get('pairs'), 'eval_set': em.get('set'), 'noise': em.get('noise', False), 'noise_profile': em.get('noise_profile', 'backend') if em.get('noise') else 'none',
                  'aer_method': em.get('backend'), 'measurement_method': em.get('measurement_method'), 'git_commit': j.get('git_commit'), 'source': os.path.basename(f)}
        if em.get('noise_profile') == 'honest': params['honest_knobs'] = HONEST
        steps = []
        for s, r in bs.items():
            s = int(s)
            steps += [('accuracy', r['accuracy'], s), ('accuracy_std_over_shot_runs', r.get('accuracy_std'), s),
                      ('accept_fraction', r.get('accept_frac_mean'), s), ('pos_fidelity_mean', r.get('pos_fidelity_mean'), s),
                      ('neg_fidelity_mean', r.get('neg_fidelity_mean'), s)]
        log(exp_name, name, f'emu:{exp_name}:{name}', params=params, steps=steps,
            tags={'analysis': 'E6 emulation', 'step_meaning': 'shots per circuit (total, incl. rejected post-selection shots)'})
print(f'added {added}, already present {skipped}')

"""Step-4 screen table: best epoch, val mrr_mean trajectory (the run's MLflow sqlite), test metrics of best.pt.
  s4_metrics.py <run_dir>...      (run dirs under checkpoints/mscoco/tn; db = mlf_dbs/<run>.db)"""
import json, sqlite3, sys
from pathlib import Path
TEST = [('coco5k_i2tR10', 'i2t@10'), ('coco5k_t2iR10', 't2i@10'), ('svo_swap_full_hard_neg_acc', 'swap_full'),
        ('svo_std_hard_neg_acc', 'svo_std'), ('aro_rel_hard_neg_acc', 'aro_rel'), ('aro_att_hard_neg_acc', 'aro_att'),
        ('sugarcrepe_replace_relation_acc', 'sc_rep_rel'), ('winoground_txt_score', 'wino_txt')]
def val_curve(run):
    db = Path(f'mlf_dbs/{run}.db')
    if not db.exists(): return {}
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    rows = con.execute("select step, value from metrics where key='mrr_mean' order by step").fetchall()
    return {s: v for s, v in rows}
print(' | '.join(['run', 'best_ep', 'val@25', 'val@50', 'val@100', 'val@last', 'val_best'] + [l for _, l in TEST]))
for d in sys.argv[1:]:
    run = Path(d).name
    try: b = json.load(open(f'{d}/benchmark_best.json')); r, ep = b['results'], b['epoch']
    except (FileNotFoundError, KeyError): r, ep = {}, None
    v = val_curve(run)
    at = lambda e: f"{v[e]:.4f}" if e in v else '-'
    last = max(v) if v else None
    print(' | '.join([run, str(ep), at(25), at(50), at(100), f"{at(last)}@{last}" if last is not None else '-',
                      f"{max(v.values()):.4f}" if v else '-'] + [f"{r[k]:.3f}" if k in r else '-' for k, _ in TEST]))

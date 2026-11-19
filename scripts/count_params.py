# Text parameter count of a config, optionally over a grid of one key, for matching baselines to the CCG-VQC (REPRODUCE.md).
#   python scripts/count_params.py -cfg configs/v1/tn.yaml --set dataset=aro --set text.n=8 ... --grid text.bond_dim=1,2,4,8
# Builds the models and initialises every train + val symbol (as train.py does) on the CPU; no training.
import argparse, copy
from modules.config import setup_exp
from modules.factory import load_experiment

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Text parameter counts of a config (over a grid of one key)")
    ap.add_argument('-cfg', '--config', required=True)
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    ap.add_argument('--grid', default=None, metavar='KEY=V1,V2,...')
    a = ap.parse_args()
    key, vals = (a.grid.split('=', 1)[0], a.grid.split('=', 1)[1].split(',')) if a.grid else (None, [None])
    for v in vals:
        sets = a.set + ([f"{key}={v}"] if key else [])
        config, _, _ = setup_exp(a.config, seed=0, overrides=sets)
        exp = load_experiment(config, 'cpu')
        n = sum(p.numel() for p in exp.text_model.parameters())
        print(f"COUNT {a.config} {' '.join(sets)} | text params {n}", flush=True)

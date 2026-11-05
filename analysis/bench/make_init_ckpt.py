# Untrained checkpoint (random-init parameters) for a bigger text.n: gives the circuit STRUCTURE for inference-only cost measurements
# (QPU transpile statistics, GPU exact-simulation energy). Weights are irrelevant to those costs. Never overwrites.
# usage: PYTHONPATH=. python analysis/bench/make_init_ckpt.py -cfg <config.yaml of an sc4 PS run> --n 7 -o <run dir>
import argparse, yaml, torch, pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp
from modules.data_pipeline.engine import DataEngine
ap = argparse.ArgumentParser(); ap.add_argument('-cfg', required=True); ap.add_argument('--n', type=int, required=True); ap.add_argument('-o', required=True)
a = ap.parse_args(); out = Path(a.o)
assert not (out / 'best.pt').exists(), f'{out}/best.pt exists; refusing to overwrite'
config, DEV, _ = setup_exp(a.cfg, seed=0, overrides=[f'text.n={a.n}', 'text.layers=2'])
ansatz, image_model, text_model, _ = build_experiment(config, 'cpu')
de = DataEngine(config, ansatz, 'cpu')
de.text_init(text_model, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True)); de.image_init(image_model)
out.mkdir(parents=True, exist_ok=True)
torch.save({'image': image_model.state_dict(), 'text': text_model.state_dict(), 'epoch': 0, 'note': 'untrained, structure only'}, out / 'best.pt')
yaml.safe_dump(config, open(out / 'config.yaml', 'w'))
print('wrote', out, 'params', sum(p.numel() for p in text_model.parameters()))

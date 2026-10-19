# Exact (no-shot) evaluation of checkpoints on: std split by negative type (subject / verb / object), the curated swap
# set, and the generated swap sets (scripts/make_swap.py). One JSON per checkpoint; never overwrites.
# usage: PYTHONPATH=. python analysis/cptp/followup/extra_eval.py <out_dir> <best.pt>...
import sys, json, copy, torch
from pathlib import Path
import pandas as pd
from modules.utils.factory import build_experiment
from modules.models.fusion.engine import MMEvaluator
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config
from modules.data_pipeline.engine import DataEngine

out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
EXTRA = {'swap_full': 'data/svo-probes/processed/swap_full.pkl', 'swap_animate': 'data/svo-probes/processed/swap_animate.pkl'}
for ck in sys.argv[2:]:
    ck = Path(ck); out = out_dir / f'{ck.parent.name}.json'
    if out.exists(): print('exists, skip', out); continue
    config, DEV, _ = setup_exp(resolve_eval_config(None, ck))
    for name, path in EXTRA.items():
        config['dataset']['test'][name] = {**copy.deepcopy(config['dataset']['test']['swap']), 'text_path': path}
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(text_model, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True)); de.image_init(image_model)
    CheckpointManager.load_model(ck, image_model, text_model, DEV); image_model.eval(); text_model.eval()
    ev = MMEvaluator(image_model, text_model, DEV)
    res = {'checkpoint': str(ck), 'discard': bool(config['text'].get('discard'))}
    for split in ['std', 'swap', *EXTRA]:
        df = de.compile_text(split); loader = de.get_loader(df, split=split)
        choice = 'image' if config['dataset']['test'][split]['task'] == 'evaluate_image_choice' else 'text'
        m, det = ev.hard_neg_eval(loader, choice=choice)
        res[split] = {'n': len(det['correct']), 'acc': m['hard_neg_acc'], 'acc_norm': m['hard_neg_acc_norm']}
        if split == 'std':
            for t in ('subj_neg', 'verb_neg', 'obj_neg'):
                mask = df[t].astype(bool).values
                c = [x for x, k in zip(det['correct'], mask) if k]
                res[f'std_{t}'] = {'n': len(c), 'acc': float(sum(c) / max(len(c), 1))}
    json.dump(res, open(out, 'w'), indent=1); print(ck.parent.name, {k: (v['acc'] if isinstance(v, dict) else v) for k, v in res.items() if k != 'checkpoint'})

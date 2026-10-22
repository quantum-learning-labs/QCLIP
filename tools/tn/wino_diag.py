"""Winoground per-item diagnostic for TN checkpoints: which items are group-correct, ties, caption-embedding similarity.
  PYTHONPATH=. python tools/tn/wino_diag.py <out.json> <ckpt.pt>..."""
import sys, json, torch
import pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine

out = {}
for ck in sys.argv[2:]:
    config, DEV, SEED = setup_exp(str(Path(ck).parent / 'config.yaml'))
    ansatz, image_model, text_model, _ = build_experiment(config, DEV)
    de = DataEngine(config, ansatz, DEV)
    de.text_init(text_model, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True))
    CheckpointManager.load_model(ck, image_model, text_model, DEV)
    text_model.to(DEV).eval(); image_model.to(DEV).eval()
    df = de.compile_text('winoground')
    text_model.extend_symbols([v for c in df.columns if c.endswith('_symbols') for v in df[c].tolist()], seed=SEED)
    S, cos_cc, same_expr = [], [], []
    with torch.no_grad():
        for batch in de.get_loader(df, split='winoground'):
            i0, i1 = image_model(batch['image_0'].to(DEV)), image_model(batch['image_1'].to(DEV))
            c0, c1 = text_model(batch['caption_0']), text_model(batch['caption_1'])
            S.append(torch.stack([(i0 * c0).sum(1), (i0 * c1).sum(1), (i1 * c0).sum(1), (i1 * c1).sum(1)], 1).cpu())
            cos_cc.append((c0 * c1).sum(1).cpu())
            same_expr += [a[0] == b[0] for a, b in zip(batch['caption_0'], batch['caption_1'])]
    S, cc = torch.cat(S), torch.cat(cos_cc)
    t = (S[:, 0] > S[:, 1]) & (S[:, 3] > S[:, 2]); i = (S[:, 0] > S[:, 2]) & (S[:, 3] > S[:, 1])
    out[ck] = {"txt": t.float().mean().item(), "img": i.float().mean().item(), "grp_items": torch.nonzero(t & i).flatten().tolist(),
               "ties": int((S[:, 0] == S[:, 1]).sum() + (S[:, 2] == S[:, 3]).sum()), "caption_cos_mean": cc.mean().item(),
               "caption_cos_gt_0.99": int((cc > 0.99).sum()), "same_einsum_structure": int(sum(same_expr)),
               "img_pair_cos_mean": None, "n": len(S),
               # caption preference independent of the image: c0 preferred by both images (or c1 by both)
               "caption_dominated": int((((S[:, 0] > S[:, 1]) == (S[:, 2] > S[:, 3]))).sum())}
    print(ck, {k: v for k, v in out[ck].items() if k != 'grp_items'}, flush=True)
ids = [set(v['grp_items']) for v in out.values()]
out['grp_items_common_to_all'] = sorted(set.intersection(*ids)) if ids else []
print("group-correct items common to all checkpoints:", len(out['grp_items_common_to_all']), "of", [len(s) for s in ids])
json.dump(out, open(sys.argv[1], 'x'), indent=1)

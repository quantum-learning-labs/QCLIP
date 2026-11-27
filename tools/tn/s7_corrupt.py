"""Step 7 headline: retrieval cost of ONE graph error per test image (image graph TN checkpoint, PSG ∩ coco5k test).
Error types (one per image, seeded; images where a type does not apply keep their graph and are reported separately):
  drop_rel  remove one relation           relabel   one relation gets another predicate      reverse  swap subject/object
  wrong_cls one object gets another class (its crop unchanged)   drop_obj  remove one object (and its relations)
  add_rel   add one relation (random object pair, random predicate)
Reports t2i (per caption) and i2t (per image) tie-neutral R@10, clean vs corrupted, on all images and on the images the
error applies to.   PYTHONPATH=. python tools/tn/s7_corrupt.py <out.json> <ckpt.pt> [seed]"""
import sys, json, random, torch
import numpy as np, pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment, load_obj
from modules.utils.general import setup_exp, CheckpointManager
from modules.data_pipeline.engine import DataEngine
from modules.models.fusion.engine import MMEvaluator
sys.path.insert(0, str(Path(__file__).parent))
from s5e_gap_breakdown_lib import encode

ck, seed = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0
config, DEV, SEED = setup_exp(str(Path(ck).parent / 'config.yaml'))
ansatz, image_model, text_model, _ = build_experiment(config, DEV)
de = DataEngine(config, ansatz, DEV)
tr, va = de.compile_text('train'), de.compile_text('val')
de.text_init(text_model, pd.concat([tr, va], ignore_index=True))
CheckpointManager.load_model(ck, image_model, text_model, DEV)
text_model.to(DEV).eval(); image_model.to(DEV).eval()
m = image_model
te = de.compile_text('coco_psg')
text_model.extend_symbols([v for c in te.columns if c.endswith('_symbols') for v in te[c].tolist() if not isinstance(v, float)], seed=SEED)
G = torch.load(config['vision'].get('graph_path', 'data/psg/graphs.pt'))
start = G['obj_start'].tolist(); n_cls, n_pred = len(G['class_names']), len(G['predicate_names'])
row_of = {int(c): i for i, c in enumerate(G['coco_id'].tolist())}
rows = [row_of[int(Path(str(f)).stem)] for f in te['file_name']]
orig = {r: m.graphs[r] for r in rows}
ev = MMEvaluator(image_model, text_model, DEV); mapper = load_obj(config['dataset']['eval_mapper'])

def recall(I, C, img_of):
    S = ev._readout(C @ I.conj().T).nan_to_num(nan=float('-inf'))   # captions x images
    io = torch.from_numpy(img_of).to(S.device)
    pos = S.gather(1, io.view(-1, 1)); neg = torch.ones_like(S, dtype=torch.bool); neg[torch.arange(len(S)), io] = False
    g = ((S > pos) & neg).sum(1).double(); t = ((S == pos) & neg).sum(1).double()
    t2i = ((10 - g) / (t + 1)).clamp(0, 1).cpu().numpy()
    St = S.T; own = torch.zeros_like(St, dtype=torch.bool); own[io, torch.arange(len(S))] = True
    best = St.masked_fill(~own, float('-inf')).amax(1, keepdim=True)
    g = ((St > best) & ~own).sum(1).double(); t = ((St == best) & ~own).sum(1).double()
    i2t = ((10 - g) / (t + 1)).clamp(0, 1).cpu().numpy()
    return t2i, i2t

def run():
    with torch.no_grad():
        I, C, img_of, _ = encode(ev, de.get_loader(te, split='coco_psg'), mapper)
    return recall(I, C, img_of), img_of

(t2i0, i2t0), img_of = run()
res = {'ckpt': ck, 'seed': seed, 'n_img': len(rows), 'clean': {'t2i_R10': float(t2i0.mean()), 'i2t_R10': float(i2t0.mean())},
       'rel_kept': [len(m.rel[r]) for r in rows]}
print('clean', res['clean'], flush=True)
cls0 = m.cls.clone()
for err in ('drop_rel', 'relabel', 'reverse', 'wrong_cls', 'drop_obj', 'add_rel'):
    rng = random.Random(seed); applied = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        objs, rels = list(range(start[r], start[r + 1])), list(m.rel[r])
        n = len(objs)
        if err in ('drop_rel', 'relabel', 'reverse') and rels:
            e = rng.randrange(len(rels)); s_, o_, p_ = rels[e]
            if err == 'drop_rel': del rels[e]
            elif err == 'relabel': rels[e] = (s_, o_, rng.choice([q for q in range(n_pred) if q != p_]))
            else: rels[e] = (o_, s_, p_)
            applied[i] = True
        elif err == 'wrong_cls' and n:
            j = objs[rng.randrange(n)]; m.cls[j] = rng.choice([c for c in range(n_cls) if c != int(cls0[j])]); applied[i] = True
            continue
        elif err == 'drop_obj' and n > 1:
            j = rng.randrange(n); objs = objs[:j] + objs[j + 1:]
            rels = [(s_ - (s_ > j), o_ - (o_ > j), p_) for s_, o_, p_ in rels if j not in (s_, o_)]; applied[i] = True
        elif err == 'add_rel' and n > 1:
            s_, o_ = rng.sample(range(n), 2); rels.append((s_, o_, rng.randrange(n_pred))); applied[i] = True
        else:
            continue
        m.set_graph(r, objs, rels)
    (t2i, i2t), _ = run()
    ai = applied; ac = applied[img_of]
    res[err] = {'n_applied': int(ai.sum()), 't2i_R10': float(t2i.mean()), 'i2t_R10': float(i2t.mean()),
                't2i_drop_applied': float((t2i0[ac] - t2i[ac]).mean()) if ac.any() else None,
                'i2t_drop_applied': float((i2t0[ai] - i2t[ai]).mean()) if ai.any() else None,
                'per_image_i2t': i2t.tolist()}
    print(err, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[err].items() if k != 'per_image_i2t'}, flush=True)
    for r in rows: m.graphs[r] = orig[r]
    m.cls.copy_(cls0)
res['clean']['per_image_i2t'] = i2t0.tolist()
json.dump(res, open(sys.argv[1], 'x'))

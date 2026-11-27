"""Step 7 data: PSG ground-truth scene graphs in the form the image graph TN reads (CPU; run once).
Writes into data/psg/:
  graphs.pt      graph rows = PSG images sorted by COCO id: coco_id, obj_start (CSR into objects), obj_cls (0..132: 80 things
                 then 53 stuff), obj_box (x1, y1, x2, y2 pixels), rel (per graph: list of (subj, obj, pred) local indices),
                 file_name, tw / tw_apex (min-degree treewidth upper bound without / with the apex node), glob (frozen CLIP
                 vector of each image from image_embeddings.pt, as the R0 baseline sees it), class / predicate names
  img_graph.pt   {coco_id: tensor([graph row])}: the dataset's image store for vision.method graph
  splits/*.pkl   our train / val / coco5k caption tables restricted to images with a PSG graph
  PYTHONPATH=. python tools/tn/s7_psg_graphs.py"""
import json, pickle, os, collections
from pathlib import Path
import numpy as np, pandas as pd, torch
from modules.utils.general import load_pkl

D = 'data/psg/'
psg = json.load(open(D + 'psg.json'))
data = sorted(psg['data'], key=lambda x: int(x['coco_image_id']))
names = psg['thing_classes'] + psg['stuff_classes']

def treewidth(n, edges):   # min-degree elimination: an upper bound (exact on trees and on small cycles)
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if a != b: adj[a].add(b); adj[b].add(a)
    alive, tw = set(range(n)), 0
    while alive:
        v = min(alive, key=lambda u: len(adj[u]))
        nb = adj[v]; tw = max(tw, len(nb))
        for a in nb:
            adj[a] |= nb - {a}; adj[a].discard(v)
        alive.discard(v); adj[v] = set()
    return tw

coco, start, cls, box, rel, fn, tw, twa = [], [0], [], [], [], [], [], []
selfloops = 0
for x in data:
    n = len(x['annotations'])
    coco.append(int(x['coco_image_id'])); fn.append(x['file_name'])
    cls += [a['category_id'] for a in x['annotations']]; box += [a['bbox'] for a in x['annotations']]
    assert all(a['bbox_mode'] == 0 for a in x['annotations'])
    r = [tuple(t) for t in x['relations'] if t[0] != t[1]]; selfloops += len(x['relations']) - len(r)
    rel.append(r); start.append(start[-1] + n)
    e = [(a, b) for a, b, _ in r]
    tw.append(treewidth(n, e)); twa.append(treewidth(n + 1, e + [(n, i) for i in range(n)]))
emb = torch.load('data/mscoco/processed/image_embeddings.pt')
missing = [c for c in coco if c not in emb]
print('graphs', len(coco), 'objects', start[-1], 'relations', sum(map(len, rel)), 'self-loops dropped', selfloops, 'no CLIP vector', len(missing))
glob = torch.stack([emb[c].float() for c in coco])
print('treewidth', sorted(collections.Counter(tw).items()), '| with apex', sorted(collections.Counter(twa).items()))
torch.save({'coco_id': torch.tensor(coco), 'obj_start': torch.tensor(start), 'obj_cls': torch.tensor(cls),
            'obj_box': torch.tensor(box, dtype=torch.float32), 'rel': rel, 'file_name': fn, 'tw': torch.tensor(tw),
            'tw_apex': torch.tensor(twa), 'glob': glob, 'class_names': names, 'predicate_names': psg['predicate_classes'],
            'test_image_ids': psg['test_image_ids']}, D + 'graphs.pt')
row = {c: i for i, c in enumerate(coco)}
torch.save({c: torch.tensor([float(i)]) for c, i in row.items()}, D + 'img_graph.pt')
os.makedirs(D + 'splits', exist_ok=True)
for src, dst in (('data/mscoco/processed/tn/train_negs.pkl', 'train_negs_psg.pkl'), ('data/mscoco/processed/tn/val_holdout.pkl', 'val_holdout_psg.pkl'),
                 ('data/mscoco/processed/validation.pkl', 'validation_psg.pkl')):
    df = load_pkl(src)
    ids = df['file_name'].map(lambda f: int(Path(str(f)).stem))
    keep = ids.isin(row)
    out = df[keep].reset_index(drop=True)
    with open(D + 'splits/' + dst, 'wb') as f: pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'{dst}: {len(out)} of {len(df)} images; columns {list(df.columns)[:6]}')
test = set(load_pkl(D + 'splits/validation_psg.pkl')['file_name'].map(lambda f: int(Path(str(f)).stem)))
trv = set(pd.concat([load_pkl(D + 'splits/train_negs_psg.pkl'), load_pkl(D + 'splits/val_holdout_psg.pkl')])['file_name'].map(lambda f: int(Path(str(f)).stem)))
print('test ∩ train/val:', len(test & trv), '| test graphs from PSG test split:', len(test & {int(i) for i in psg['test_image_ids']}) if psg['test_image_ids'] and str(psg['test_image_ids'][0]).isdigit() else 'n/a')

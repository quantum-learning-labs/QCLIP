"""Data for the classical TN benchmarks (configs/tn/). Writes new files only: a step whose outputs exist is skipped.

  coco_split  MSCOCO train2017 -> tn/train.pkl + tn/val_holdout.pkl (5k images, seed 0); val2017 stays the test set
  coco_clip   CLIP ViT-B/32 vectors of train2017 + val2017 -> data/mscoco/processed/image_embeddings.pt
  sugarcrepe  SugarCREPE++ (5 subsets): HF download, Bobcat parse -> data/sugarcrepe/processed/<subset>.pkl
  winoground  Winoground (gated on HF: needs `huggingface-cli login`): parse, images -> raw/images.zip + CLIP .pt

  PYTHONPATH=. python scripts/prep_tn_data.py coco_split coco_clip sugarcrepe winoground
"""
import argparse, io, json, os, pickle, zipfile
from pathlib import Path
import numpy as np
import torch

from modules.utils.general import get_device, load_pkl

DATA = Path('data')
PARSER_PATH = '/cs/research/pplv/comp_bridge/bobcat'

def write_new(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as f:   # never overwrite
        pickle.dump(obj, f)
    print(f" wrote {path}")

def coco_split(n_holdout=5000, seed=0):
    out = DATA / 'mscoco/processed/tn'
    if (out / 'val_holdout.pkl').exists(): return print(" coco_split: exists, skipped")
    df = load_pkl(DATA / 'mscoco/processed/train.pkl')
    perm = np.random.default_rng(seed).permutation(len(df))
    hold = np.sort(perm[:n_holdout])
    write_new(df.iloc[hold].reset_index(drop=True), out / 'val_holdout.pkl')
    write_new(df.drop(df.index[hold]).reset_index(drop=True), out / 'train.pkl')

def coco_clip(device):
    from modules.data_pipeline.embeddings import ImgStream, embed_images
    final = DATA / 'mscoco/processed/image_embeddings.pt'
    if final.exists(): return print(" coco_clip: exists, skipped")
    parts = []
    for name in ('train2017', 'val2017'):
        part = DATA / f'mscoco/processed/clip_{name}.pt'
        if not part.exists():
            embed_images(ImgStream(str(DATA / f'mscoco/raw/images/{name}.zip'), file_type='zip'), str(part), device=device)
        parts.append(torch.load(part))
    merged = {k: v for p in parts for k, v in p.items()}
    assert len(merged) == sum(len(p) for p in parts), "image ids collide between train2017 and val2017"
    torch.save(merged, final)
    print(f" wrote {final} ({len(merged)} images)")

# gated on HF: read the parquet of the downloaded snapshot directly
WINOGROUND_PARQUET = DATA / '.hf-cache/hub/datasets--facebook--winoground/snapshots/b400e173549071916ad1b3d449293bc8d8b4b763/data/test-00000-of-00001.parquet'

def parse(name, device, data=None):
    # HF download (BaseDatasetRetriever) + Bobcat parse (TextProcessor) into data/<name>/processed/_parsed/<split>.pkl
    from modules.data_pipeline.retrieval import BaseDatasetRetriever, DATASET_SOURCES
    from modules.symbolic.text_processor import TextProcessor
    if data is None:
        retriever = BaseDatasetRetriever(name, data_root=str(DATA))
        retriever.retrieve()
        data = retriever.data
    text_labels = DATASET_SOURCES[name]['text_label']
    parsed = DATA / name / 'processed/_parsed'
    todo = {k: v for k, v in data.items() if not (parsed / f'{k}.pkl').exists()}
    for k, v in data.items():
        print(f" {name}/{k}: {len(v)} rows, columns {list(v.column_names if hasattr(v, 'column_names') else v.columns)}")
    if todo:
        TextProcessor(PARSER_PATH, 64, device).text2diagram(path=parsed, dataset=todo, text_labels=text_labels)
    stats = {}
    for k, v in data.items():
        n_parsed = len(load_pkl(parsed / f'{k}.pkl'))
        stats[k] = {"rows": len(v), "parsed": n_parsed, "coverage": n_parsed / len(v)}
    print(f" {name} parse coverage: {stats}")
    return parsed, stats

def sugarcrepe(device):
    out = DATA / 'sugarcrepe/processed'
    parsed, stats = parse('sugarcrepe', device)
    for f in sorted(parsed.glob('*.pkl')):
        if not (out / f.name).exists():
            write_new(load_pkl(f), out / f.name)   # images: filename in the COCO val2017 zip / CLIP .pt
    if not (out / 'parse_stats.json').exists():
        json.dump(stats, open(out / 'parse_stats.json', 'x'), indent=1)

def winoground(device):
    from modules.data_pipeline.embeddings import ImgStream, embed_images
    out = DATA / 'winoground/processed'
    import pandas as pd
    raw = pd.read_parquet(WINOGROUND_PARQUET)
    parsed, stats = parse('winoground', device, data={'test': raw[['id', 'image_0', 'image_1', 'caption_0', 'caption_1', 'tag']]})
    zpath = DATA / 'winoground/raw/images.zip'
    df = load_pkl(parsed / 'test.pkl')
    if not zpath.exists():
        # image k of example id -> winoground/<2*id+k>.png (int names: the .pt store and the zip loader key on them)
        with zipfile.ZipFile(zpath, 'x') as z:
            z.writestr('winoground/', '')
            for _, row in df.iterrows():
                for k in (0, 1):
                    img = row[f'image_{k}']
                    data = img['bytes'] if isinstance(img, dict) else img
                    if not isinstance(data, (bytes, bytearray)):
                        buf = io.BytesIO(); img.save(buf, format='PNG'); data = buf.getvalue()
                    z.writestr(f"winoground/{2 * int(row['id']) + k}.png", data)
        print(f" wrote {zpath}")
    if not (out / 'test.pkl').exists():
        df = df.copy()
        for k in (0, 1):
            df[f'image_{k}'] = 2 * df['id'].astype(int) + k
        write_new(df, out / 'test.pkl')
    if not (out / 'image_embeddings.pt').exists():
        embed_images(ImgStream(str(zpath), file_type='zip'), str(out / 'image_embeddings.pt'), device=device)
    if not (out / 'parse_stats.json').exists():
        json.dump(stats, open(out / 'parse_stats.json', 'x'), indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('steps', nargs='+', choices=['coco_split', 'coco_clip', 'sugarcrepe', 'winoground'])
    args = ap.parse_args()
    os.environ.setdefault("HF_HOME", str((DATA / '.hf-cache').resolve()))
    dev = get_device()
    for step in args.steps:
        print(f"[{step}]")
        try:
            coco_split() if step == 'coco_split' else globals()[step](dev)
        except Exception as e:   # one missing dataset (e.g. gated Winoground) must not stop the others
            print(f" {step} FAILED: {type(e).__name__}: {e}")

# Writes the train/val/test assignment of every pool row, and each test set's rows, for the given split seeds, so the
# splits of the v1 results can be checked (or used) without this code:
#   python scripts/export_splits.py --dataset svo-probes --seeds 0 1 2 3 4 -o splits/svo-probes
# splits/<dataset>/seed<k>.csv: one line per pool row (row = position in data/<dataset>/processed/all.pkl), its group key,
# split label and identifying columns, and a 0/1 column per test set (in_<test set>; swap sets: the row's caption is in it).
# splits/<dataset>/records.json: the split record (spec, id, sizes) of each seed and the md5 of the pool file.
import argparse, hashlib, json, os
import modules.data_pipeline.engine as engine_mod
from modules.config import setup_exp
from modules.data_pipeline.engine import DataEngine

ID_COLS = {'svo-probes': ['pos_image_id', 'neg_image_id', 'corrected_sentence'],
           'aro': ['subset', 'vg_image_id', 'image_id']}   # captions: the row of all.pkl

def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export the v1 splits as CSV")
    ap.add_argument('--dataset', required=True, choices=list(ID_COLS))
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('-o', '--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    load = engine_mod.load_pkl
    def load_with_row(path):   # tag every pool row with its position, so it survives select / derive / sample
        df = load(path)
        if 'pool_row' not in df.columns:
            df = df.assign(pool_row=range(len(df)))
        return df
    engine_mod.load_pkl = load_with_row

    records = {}
    for seed in args.seeds:
        config, _, _ = setup_exp('configs/vqc.yaml', overrides=[f'dataset={args.dataset}', f'dataset.split.seed={seed}'])
        engine = DataEngine(config, None, 'cpu')
        pool_path = config['dataset']['train']['text_path']
        pool = load_with_row(pool_path)
        labels, record = engine_mod.assign(pool, config['dataset']['split'])
        out = pool[['pool_row', config['dataset']['split']['by']] + [c for c in ID_COLS[args.dataset] if c != config['dataset']['split']['by']]].copy()
        out.insert(2, 'split', labels)
        for test_name in config['dataset']['test']:
            rows = set(engine.load_text(test_name)['pool_row'])
            out[f'in_{test_name}'] = out['pool_row'].isin(rows).astype(int)
        out.to_csv(os.path.join(args.out, f'seed{seed}.csv'), index=False)
        records[f'seed{seed}'] = record
        print(seed, record['id'], record['rows'], {c: int(out[c].sum()) for c in out.columns if c.startswith('in_')})
    records['pool'] = {'path': pool_path, 'md5': md5(pool_path), 'rows': len(pool)}
    json.dump(records, open(os.path.join(args.out, 'records.json'), 'w'), indent=1)

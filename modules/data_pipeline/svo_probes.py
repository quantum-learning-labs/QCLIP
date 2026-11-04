"""SVO-Probes from code only: metadata from DeepMind's public CSV, images from its URLs, our splits from a small manifest.

What is downloaded: svo_probes.csv (github.com/google-deepmind/svo_probes) and the images behind its pos_url / neg_url.
What ships in this repo (svo_probes_assets/): split_manifest.csv.gz (which CSV row is in train/val/test, plus the corrected
caption) and swap.pkl.gz (the 95 hand-written subject/object-swap pairs, which are not part of the DeepMind release).
Image ids are the CSV's own pos_image_id / neg_image_id, stored as images/<id>.jpg in raw/images.zip.
A few image links have died since the release: failures are listed in raw/images_missing.csv and the affected rows are
dropped from the in-memory data by BaseDatasetRetriever (the raw pickles stay complete).
"""
from __future__ import annotations
import io, zipfile, urllib.request, urllib.error, concurrent.futures as cf, time
from pathlib import Path
import pandas as pd
from PIL import Image

CSV_URL = "https://raw.githubusercontent.com/google-deepmind/svo_probes/main/svo_probes.csv"
ASSETS = Path(__file__).resolve().parent / "svo_probes_assets"
SPLITS = ("train", "val", "test")
COLS = ['sentence', 'corrected_sentence', 'pos_triplet', 'neg_triplet', 'pos_url', 'neg_url', 'pos_image_id', 'neg_image_id',
        'subj_neg', 'verb_neg', 'obj_neg', 'subj', 'verb', 'obj']
UA = {"User-Agent": "Mozilla/5.0 (compatible; QCLIP-data-fetch)"}


def _download(url: str, timeout: int = 30, tries: int = 3) -> bytes | None:
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410): return None           # gone for good
        except Exception:
            pass
        time.sleep(1 + t)
    return None


def fetch_csv(raw_dir: Path) -> pd.DataFrame:
    f = raw_dir / "svo_probes.csv"
    if not f.exists():
        data = _download(CSV_URL)
        if data is None: raise RuntimeError(f"could not download {CSV_URL}")
        f.write_bytes(data)
    return pd.read_csv(f)


def build_splits(csv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """train/val/test in exactly the layout of the original raw pickles (index, columns, dtypes)."""
    man = pd.read_csv(ASSETS / "split_manifest.csv.gz")
    out = {}
    for s in SPLITS:
        m = man[man.split == s]
        df = csv.iloc[m.dm_row.values].reset_index(drop=True).copy()
        df['corrected_sentence'] = m.corrected_sentence.values
        t = df.pos_triplet.str.split(',', expand=True)
        df['subj'], df['verb'], df['obj'] = t[0], t[1], t[2]
        out[s] = df[COLS]
        out[s].index = pd.Index(m['index'].values)
    out['swap'] = pd.read_pickle(ASSETS / "swap.pkl.gz")
    return out


def image_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = [d[[c, k]].set_axis(['url', 'id'], axis=1) for d in splits.values() for c, k in (('pos_url', 'pos_image_id'), ('neg_url', 'neg_image_id'))]
    return pd.concat(parts).drop_duplicates('id').sort_values('id').reset_index(drop=True)


def fetch_images(table: pd.DataFrame, zip_path: Path, workers: int = 32, size_limit: int | None = None) -> pd.DataFrame:
    """Download every image id in `table` into zip_path as images/<id>.jpg (RGB JPEG). Resumable: ids already in the zip are
    skipped. Returns the table of ids that could not be fetched (also written next to the zip as images_missing.csv)."""
    have = set()
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z: have = {n for n in z.namelist() if n.endswith('.jpg')}
    todo = table[~table.id.map(lambda i: f"images/{i}.jpg" in have)]
    def one(r):
        raw = _download(r.url)
        if raw is None: return r.id, None
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            if size_limit: im.thumbnail((size_limit, size_limit))
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=92); return r.id, buf.getvalue()
        except Exception:
            return r.id, None
    failed = []
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_STORED) as z:
        if "images/" not in z.namelist(): z.writestr("images/", b"")     # BaseDataset reads the folder name from entry 0
        with cf.ThreadPoolExecutor(workers) as ex:
            for n, (i, b) in enumerate(ex.map(one, list(todo.itertuples())), 1):
                if b is None: failed.append(i)
                else: z.writestr(f"images/{i}.jpg", b)
                if n % 500 == 0: print(f"  images {n}/{len(todo)} ({len(failed)} missing)", flush=True)
    missing = table[table.id.isin(failed)]
    # ids that failed before and are not in the zip now are still missing
    with zipfile.ZipFile(zip_path) as z: names = set(z.namelist())
    missing = table[~table.id.map(lambda i: f"images/{i}.jpg" in names)]
    missing.to_csv(zip_path.parent / "images_missing.csv", index=False)
    return missing


def fetch(raw_dir: Path, images: bool = True, workers: int = 32, csv_path: Path | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Make <raw_dir> hold train/val/test/swap.pkl (+ images.zip). Never overwrites existing files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv = pd.read_csv(csv_path) if csv_path else fetch_csv(raw_dir)
    splits = build_splits(csv)
    for s, d in splits.items():
        f = raw_dir / f"{s}.pkl"
        if not f.exists(): d.to_pickle(f)
    missing = pd.DataFrame(columns=['url', 'id'])
    if images:
        missing = fetch_images(image_table(splits), raw_dir / "images.zip", workers)
    return splits, missing


def drop_missing(splits: dict[str, pd.DataFrame], missing: pd.DataFrame) -> dict[str, pd.DataFrame]:
    bad = set(missing.id)
    return {s: d[~(d.pos_image_id.isin(bad) | d.neg_image_id.isin(bad))] for s, d in splits.items()}

from __future__ import annotations
import os, io, json, time, pickle, hashlib, zipfile
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve, urlopen, Request
from urllib.error import HTTPError
from datasets import load_dataset

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests"

DATASET_SOURCES = {
    "svo-probes": {'csv_url': 'https://raw.githubusercontent.com/google-deepmind/svo_probes/main/svo_probes.csv',
                   'manifest': 'svo-probes',
                   'columns': ['sentence', 'corrected_sentence', 'pos_triplet', 'neg_triplet', 'pos_url', 'neg_url', 'pos_image_id',
                               'neg_image_id', 'subj_neg', 'verb_neg', 'obj_neg', 'subj', 'verb', 'obj'],
                   'split_columns': {'pos_triplet': ['subj', 'verb', 'obj']},
                   'img_label': ['pos_image_id', 'neg_image_id'],
                   'img_url_label': ['pos_url', 'neg_url'],
                   'text_label': ['corrected_sentence']},
}

class BaseDatasetRetriever():
    def __init__(self, dataset_name: str, data_root: str | Path = DEFAULT_DATA_ROOT):
        if DATASET_SOURCES.get(dataset_name) is None:
            raise ValueError(f"Dataset {dataset_name} is not supported. Please choose from: {list(DATASET_SOURCES.keys())}")
        
        self.dataset_name = dataset_name
        self.img_labels = DATASET_SOURCES[dataset_name]['img_label']
        self.csv_url = DATASET_SOURCES[dataset_name].get('csv_url', None)
        self.manifest_dir = MANIFEST_ROOT / DATASET_SOURCES[dataset_name]['manifest'] if self.csv_url is not None else None
        self.columns = DATASET_SOURCES[dataset_name].get('columns', None)
        self.split_columns = DATASET_SOURCES[dataset_name].get('split_columns', {})
        self.img_url_labels = DATASET_SOURCES[dataset_name].get('img_url_label', None)

        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / self.dataset_name / "raw"
        self.images_dir = None
        # the raw images (zip of images/<image id>.jpg) that scripts/retrieve.py embeds
        self.images_zip = self.raw_dir / DATASET_SOURCES[dataset_name].get('images_file', 'images.zip')
        self.data = {}   # filled by retrieve(): split -> rows ('all' for the pools we split ourselves)

        self._set_hf_cache()

    def _set_hf_cache(self) -> None:
        cache_root = (self.data_root / ".hf-cache").resolve()
        os.environ.setdefault("HF_HOME", str(cache_root))

    def retrieve(self, fetch_images: bool = True, workers: int = 32):
        """Unified execution execution method with validation checks."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.data = {}
        self._retrieve_csv()
        self._retrieve_image_urls(workers)
        return self.data

    def _retrieve_csv(self):
        csv_path = self.raw_dir / self.csv_url.split('/')[-1]
        if not os.path.exists(csv_path):
            urlretrieve(self.csv_url, csv_path)
        csv = pd.read_csv(csv_path)

        rows = pd.read_csv(self.manifest_dir / "curated.csv.gz")
        df = csv.iloc[rows['csv_row'].values].reset_index(drop=True)
        for col in rows.columns.drop('csv_row'):
            df[col] = rows[col].values
        for col, new_cols in self.split_columns.items():
            parts = df[col].str.split(',', expand=True)
            for i, new_col in enumerate(new_cols):
                df[new_col] = parts[i]
        self.data['all'] = df[self.columns + ['legacy_split']] if self.columns is not None else df
        self._save_raw()

    def _save_raw(self):
        for split, df in self.data.items():
            fpath = self.raw_dir / f"{split}.pkl"
            if not os.path.exists(fpath):
                df.to_pickle(fpath)

    def _retrieve_image_urls(self, workers: int = 32):
        zip_path = self.images_zip
        parts = [df[[url_col, id_col]].set_axis(['url', 'id'], axis=1)
                 for df in self.data.values() for url_col, id_col in zip(self.img_url_labels, self.img_labels)]
        images = pd.concat(parts).drop_duplicates('id').sort_values('id').reset_index(drop=True)

        stored = set()
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path) as archive:
                stored = set(archive.namelist())
        todo = images[~images['id'].map(lambda i: f"images/{i}.jpg" in stored)]
        with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_STORED) as archive:
            if "images/" not in archive.namelist():
                archive.writestr("images/", b"")   # BaseDataset reads the folder name from entry 0
            with ThreadPoolExecutor(workers) as pool:
                for n, (img_id, img_bytes) in enumerate(pool.map(download_image, todo.itertuples()), 1):
                    if img_bytes is not None:
                        archive.writestr(f"images/{img_id}.jpg", img_bytes)
                    if n % 500 == 0:
                        print(f"  images {n}/{len(todo)}", flush=True)
            stored = set(archive.namelist())

        missing = images[~images['id'].map(lambda i: f"images/{i}.jpg" in stored)]
        missing.to_csv(self.raw_dir / "images_missing.csv", index=False)
        if len(missing):
            print(f"{self.dataset_name}: {len(missing)} image links are dead (listed in {self.raw_dir / 'images_missing.csv'}); "
                  f"rows using them are dropped from retriever.data")
            bad = set(missing['id'])
            for split, df in self.data.items():
                keep = ~df[self.img_labels].isin(bad).any(axis=1)
                self.data[split] = df[keep]

def download_image(row, timeout: int = 30, tries: int = 3):
    # (id, RGB JPEG bytes), or (id, None) if the link is dead
    for t in range(tries):
        try:
            with urlopen(Request(row.url, headers={"User-Agent": "Mozilla/5.0 (compatible; QCLIP-data-fetch)"}), timeout=timeout) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=92)
            return row.id, buf.getvalue()
        except HTTPError as e:
            if e.code in (403, 404, 410):
                return row.id, None
        except Exception:
            pass
        time.sleep(1 + t)
    return row.id, None

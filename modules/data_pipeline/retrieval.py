from __future__ import annotations
import os, io, time, pickle, zipfile
import pandas as pd
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlretrieve, urlopen, Request
from urllib.error import HTTPError
from datasets import load_dataset

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
# our splits of datasets distributed as a single CSV: <name>/splits.csv.gz (split, index, csv_row + columns that replace
# or add CSV columns) and extra splits as <split>.pkl.gz
MANIFEST_ROOT = Path(__file__).resolve().parent / "manifests"

DATASET_SOURCES = {
    "winoground": {'url': 'facebook/winoground', 
                   'img_label': ['image_0', 'image_1'], 
                   'text_label': ['caption_0', 'caption_1']},
    "sugarcrepe": {'url': 'Aman-J/SugarCrepe_pp', 
                   'subset': ["swap_object", "swap_atribute", "replace_object", "replace_attribute", "replace_relation"],
                   'img_label': ['filename'],
                   'text_label': ['caption', 'negative_caption', 'caption2'],
                   'img_dataset_name': 'mscoco',
                   'img_url': ['http://images.cocodataset.org/zips/train2017.zip', 'http://images.cocodataset.org/zips/val2017.zip']},
    "mscoco": {'url': 'phiyodr/coco2017', 
               'img_label': ['image_id'], 
               'text_label': ['captions'],
               'img_url': ['http://images.cocodataset.org/zips/train2017.zip', 'http://images.cocodataset.org/zips/val2017.zip']},
    "aro-att": {'url': 'gowitheflow/ARO-Visual-Attribution', 
                'img_label': ['image'],
                'text_label': ['true_caption', 'false_caption']},
    "aro-rel": {'url': 'gowitheflow/ARO-Visual-Relation', 
                'img_label': ['image'],
                'text_label': ['true_caption', 'false_caption']}, 
    "svo-probes": {'csv_url': 'https://raw.githubusercontent.com/google-deepmind/svo_probes/main/svo_probes.csv',
                   'manifest': 'svo-probes',
                   'columns': ['sentence', 'corrected_sentence', 'pos_triplet', 'neg_triplet', 'pos_url', 'neg_url', 'pos_image_id',
                               'neg_image_id', 'subj_neg', 'verb_neg', 'obj_neg', 'subj', 'verb', 'obj'],
                   'split_columns': {'pos_triplet': ['subj', 'verb', 'obj']},
                   'img_label': ['pos_image_id', 'neg_image_id'],
                   'img_url_label': ['pos_url', 'neg_url'],
                   'text_label': ['corrected_sentence']},
    "aro": {'img_label': ['image_id'],
            'text_label': ['true_caption', 'false_caption']},
}

class BaseDatasetRetriever():
    def __init__(self, dataset_name: str, data_root: str | Path = DEFAULT_DATA_ROOT):
        if DATASET_SOURCES.get(dataset_name) is None:
            raise ValueError(f"Dataset {dataset_name} is not supported. Please choose from: {list(DATASET_SOURCES.keys())}")
        
        self.dataset_name = dataset_name
        self.img_dataset_name = DATASET_SOURCES[dataset_name].get('img_dataset_name', dataset_name)
        self.dataset_url = DATASET_SOURCES[dataset_name].get('url', None)
        self.subsets = DATASET_SOURCES[dataset_name].get('subset', None)
        self.img_url = DATASET_SOURCES[dataset_name].get('img_url', None)
        self.img_labels = DATASET_SOURCES[dataset_name]['img_label']
        self.text_labels = DATASET_SOURCES[dataset_name]['text_label']
        self.csv_url = DATASET_SOURCES[dataset_name].get('csv_url', None)
        self.manifest_dir = MANIFEST_ROOT / DATASET_SOURCES[dataset_name]['manifest'] if self.csv_url is not None else None
        self.columns = DATASET_SOURCES[dataset_name].get('columns', None)
        self.split_columns = DATASET_SOURCES[dataset_name].get('split_columns', {})
        self.img_url_labels = DATASET_SOURCES[dataset_name].get('img_url_label', None)

        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / self.dataset_name / "raw"
        if self.img_url is not None:
            self.images_dir = self.data_root / self.img_dataset_name / "raw" / "images"
        else:
            self.images_dir = None
        self.data = {}

        self._set_hf_cache()

        if os.path.exists(self.raw_dir):
            files = list(self.raw_dir.glob("*.pkl"))
            for file in files:
                with open(file, "rb") as f:
                    data = pickle.load(f)
                self.data[str(file).split('.')[0].split('/')[-1]] = data

    def _set_hf_cache(self) -> None:
        cache_root = (self.data_root / ".hf-cache").resolve()
        os.environ.setdefault("HF_HOME", str(cache_root))

    def retrieve(self, fetch_images: bool = True, workers: int = 32):
        """Unified execution execution method with validation checks."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.csv_url is not None:
            self._retrieve_csv()
        else:
            self._retrieve_hf()

        if fetch_images and self.img_url_labels is not None:
            self._retrieve_image_urls(workers)
        elif fetch_images and self.img_url is not None:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            for url in self.img_url:
                zip_name = url.split('/')[-1]
                zip_target = self.images_dir / zip_name
                if not os.path.exists(zip_target):
                    urlretrieve(url, zip_target)
        return self.data

    def _retrieve_hf(self):
        ds = {}
        if self.subsets is not None:
            for subset in self.subsets:
                tmp_ds = load_dataset(self.dataset_url, subset, cache_dir=str(self.data_root / ".hf-cache")) 
                if len(tmp_ds) == 1:
                    ds[subset] = tmp_ds.values()[0]
                else:
                    raise ValueError(f"Expected a single split for subset {subset}, but got {len(tmp_ds)} splits.")
            splits = self.subsets
        else:
            ds = load_dataset(self.dataset_url, cache_dir=str(self.data_root / ".hf-cache"))
            splits = list(ds.keys())

        for split in splits:
            self.data[split] = ds[split]
            fpath = self.raw_dir / f"{split}.pkl"
            if not os.path.exists(fpath):
                with open(fpath, "wb") as f:
                    pickle.dump(ds[split], f)

    def _retrieve_csv(self):
        csv_path = self.raw_dir / self.csv_url.split('/')[-1]
        if not os.path.exists(csv_path):
            urlretrieve(self.csv_url, csv_path)
        csv = pd.read_csv(csv_path)

        manifest = pd.read_csv(self.manifest_dir / "splits.csv.gz")
        for split, rows in manifest.groupby('split', sort=False):
            df = csv.iloc[rows['csv_row'].values].reset_index(drop=True)
            for col in rows.columns.drop(['split', 'index', 'csv_row']):
                df[col] = rows[col].values
            for col, new_cols in self.split_columns.items():
                parts = df[col].str.split(',', expand=True)
                for i, new_col in enumerate(new_cols):
                    df[new_col] = parts[i]
            df = df[self.columns] if self.columns is not None else df
            df.index = pd.Index(rows['index'].values)
            self.data[split] = df
        for fpath in self.manifest_dir.glob("*.pkl.gz"):
            self.data[fpath.name.split('.')[0]] = pd.read_pickle(fpath)

        for split, df in self.data.items():
            fpath = self.raw_dir / f"{split}.pkl"
            if not os.path.exists(fpath):
                df.to_pickle(fpath)

    def _retrieve_image_urls(self, workers: int = 32):
        # images listed by URL in the rows: stored as images/<id>.jpg in raw/images.zip (resumable). Links that have died since
        # the release are listed in raw/images_missing.csv; rows using them are dropped from self.data (raw pickles stay complete)
        zip_path = self.raw_dir / "images.zip"
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

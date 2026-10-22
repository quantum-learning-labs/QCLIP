from __future__ import annotations
import os, pickle
from pathlib import Path
from urllib.request import urlretrieve
from datasets import load_dataset

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

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
    "svo-probes": {'img_label': ['pos_image_id', 'neg_image_id'],
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

    def retrieve(self):
        """Unified execution execution method with validation checks."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_url is None:   # local-only datasets (svo-probes, aro): raw pickles must already be in raw/
            if not self.data:
                raise FileNotFoundError(f"{self.dataset_name} has no download source and {self.raw_dir} has no .pkl files")
            return self.data
        ds = {}
        if self.subsets is not None:
            for subset in self.subsets:
                tmp_ds = load_dataset(self.dataset_url, subset, cache_dir=str(self.data_root / ".hf-cache")) 
                if len(tmp_ds) == 1:
                    ds[subset] = list(tmp_ds.values())[0]
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

        if self.img_url is not None:
            self.images_dir.mkdir(parents=True, exist_ok=True)
            for url in self.img_url:
                zip_name = url.split('/')[-1]
                zip_target = self.images_dir / zip_name
                if not os.path.exists(zip_target):
                    urlretrieve(url, zip_target)
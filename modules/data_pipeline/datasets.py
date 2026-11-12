from __future__ import annotations
import zipfile, io, torch, random, re
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, compiled_df: pd.DataFrame, img_path: str | None, image_transform=None, mode: str = "train"):
        self.compiled_df = compiled_df
        self.image_transform = image_transform
        self.img_path = img_path
        self.img_store = img_path.split('.')[-1].lower() if img_path is not None else 'df'
        self.mode = mode.lower()

        if self.img_store == 'zip':
            with zipfile.ZipFile(self.img_path, 'r') as archive:
                # self.archive = zipfile.ZipFile(img_path, 'r')
                self.namelist_set = set(archive.namelist())
                self.parent = archive.namelist()[0]
                self.suffix = Path(archive.namelist()[1]).suffix
            self.archive = None
        elif self.img_store == 'pt':
            self.embeddings = torch.load(img_path)

    def __len__(self):
        return len(self.compiled_df) 

    def _load_image(self, image_input: str) -> Image.Image:
        if pd.isna(image_input) or image_input is None:
            raise ValueError("Encountered empty or missing image identifier reference row.")
        
        if self.img_store == 'pt':
            return self.embeddings[int(image_input)]
        elif self.img_store == 'df':
            img = image_input.convert("RGB")
        elif self.img_store == 'zip':
            if self.archive is None:
                self.archive = zipfile.ZipFile(self.img_path, 'r')
            zip_path = str(image_input)
            if not zip_path.startswith(self.parent):
                zip_path = self.parent + zip_path
            if not zip_path.endswith(self.suffix):
                zip_path += self.suffix
            img_bytes = self.archive.read(zip_path)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            raise ValueError(f"Unsupported layout configuration '{self.img_store}'.")

        if self.image_transform:
            img = self.image_transform(img)
        return img

class CocoDataset(BaseDataset):
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load the single image reference
        image_ref = row_compiled.get('file_name')
        image = self._load_image(image_ref)
        
        # Compiled data contains lists for columns with multiple captions
        einsums = row_compiled['captions_einsum']
        symbols = row_compiled['captions_symbols']
        
        # During contrastive training, pick one caption variation randomly per sample
        if self.mode == "train":
            caption_idx = random.randint(0, len(einsums) - 1)
        
            return {"image": image,
                    "caption": (einsums[caption_idx], symbols[caption_idx])
                    }
        else:
            return {"image": image,
                    "caption": list(zip(einsums, symbols))
                    }

def coco_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch] 
    
    return {
        "image": images,
        "caption": captions
    }

class SVODataset(BaseDataset):
    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        
        # Load image
        pos_image = self._load_image(row_compiled['pos_image_id'])
        neg_image = self._load_image(row_compiled['neg_image_id'])
        
        # Extract both expressions side-by-side for comparison scoring
        einsum = row_compiled['corrected_sentence_einsum']
        symbols = row_compiled['corrected_sentence_symbols']
        
        item = {"caption": (einsum, symbols),
                "pos_image": pos_image,
                "neg_image": neg_image,
                "idx": idx,   # row index (distillation looks up the teacher's embeddings by it)
                }
        if 'swapped_sentence_einsum' in row_compiled.index:   # training with swapped-caption negatives (swap_negatives)
            item["swap_caption"] = (row_compiled['swapped_sentence_einsum'], row_compiled['swapped_sentence_symbols'])
            item["swap_valid"] = bool(row_compiled.get('swap_valid', True))
        return item

def svo_collate_fn(batch):
    captions = [item['caption'] for item in batch]
    pos_images = torch.stack([item['pos_image'] for item in batch])
    neg_images = torch.stack([item['neg_image'] for item in batch])
    
    return {
        "caption": captions,
        "pos_image": pos_images,
        "neg_image": neg_images,
        "idx": torch.tensor([item['idx'] for item in batch]) if 'idx' in batch[0] else None,
        **({"swap_caption": [item['swap_caption'] for item in batch],
            "swap_valid": torch.tensor([item['swap_valid'] for item in batch])} if 'swap_caption' in batch[0] else {}),
    }

class TextChoiceDataset(BaseDataset):
    # one image and two captions, the true one first (SVO swap sets: subject and object swapped; ARO: attributes swapped)
    image_col, pos_col, neg_col = 'pos_image_id', 'corrected_sentence', 'swapped_sentence'

    def __getitem__(self, idx):
        row_compiled = self.compiled_df.iloc[idx]
        return {"image": self._load_image(row_compiled[self.image_col]),
                "pos_caption": (row_compiled[f'{self.pos_col}_einsum'], row_compiled[f'{self.pos_col}_symbols']),
                "neg_caption": (row_compiled[f'{self.neg_col}_einsum'], row_compiled[f'{self.neg_col}_symbols'])}

class SwapDataset(TextChoiceDataset):
    pass

class ARODataset(TextChoiceDataset):
    image_col, pos_col, neg_col = 'image_id', 'true_caption', 'false_caption'

def text_choice_collate_fn(batch):
    return {
        "image": torch.stack([item['image'] for item in batch]),
        "pos_caption": [item['pos_caption'] for item in batch],
        "neg_caption": [item['neg_caption'] for item in batch]
    }

# verbs whose subject/object swap usually still describes the image: their swapped captions are not negatives
SYMMETRIC_VERBS = {'meet', 'hug', 'kiss', 'marry', 'touch', 'fight', 'play', 'talk', 'dance', 'race', 'wrestle', 'greet',
                   'shake', 'face', 'resemble', 'join', 'accompany', 'chat', 'argue', 'embrace', 'cuddle', 'date'}

def build_swap_set(df, keep_all=False):
    # SwapDataset layout from a processed SVO split: for each caption whose diagram has exactly two noun boxes (type ['n']),
    # swap the two nouns (same grammar and vocabulary, so no new symbols). Rows without a real swap (not two nouns, the
    # same noun twice, a symmetric verb) are dropped, or with keep_all kept with swap_valid False (swapped-caption
    # negatives for training: the training set itself is unchanged)
    rows = []
    stats = {'total': len(df), 'not_two_nouns': 0, 'same_noun': 0, 'symmetric': 0}
    for _, row in df.iterrows():
        diagram = row['corrected_sentence_diagram']
        nouns = [i for i, box in enumerate(diagram) if list(box[2]) == ['n']]
        if len(nouns) != 2:
            reason = 'not_two_nouns'
        elif diagram[nouns[0]][0].lower() == diagram[nouns[1]][0].lower():
            reason = 'same_noun'
        elif str(row['verb']).lower() in SYMMETRIC_VERBS:
            reason = 'symmetric'
        else:
            reason = None

        new_row = row.to_dict()
        if reason is not None:
            stats[reason] += 1
            if not keep_all:
                continue
            new_row['swapped_sentence'] = row['corrected_sentence']
            new_row['swapped_sentence_diagram'] = diagram
            new_row['swap_valid'] = False
        else:
            a, b = nouns
            word_a, word_b = diagram[a][0], diagram[b][0]
            swapped = [list(box) for box in diagram]
            swapped[a][0], swapped[b][0] = word_b, word_a
            swapped = [type(box)(new) if isinstance(box, tuple) else new for box, new in zip(diagram, swapped)]   # keep the box type
            sentence = re.sub(rf'\b{re.escape(word_a)}\b', '\0', row['corrected_sentence'], count=1)
            sentence = re.sub(rf'\b{re.escape(word_b)}\b', word_a, sentence, count=1).replace('\0', word_b)
            new_row['swapped_sentence'] = sentence
            new_row['swapped_sentence_diagram'] = swapped
            new_row['swap_valid'] = True
        rows.append(new_row)

    swap_df = pd.DataFrame(rows)
    cols = [c for c in swap_df.columns if not c.endswith('_diagram')] + ['corrected_sentence_diagram', 'swapped_sentence_diagram']
    print(f"Swap set: {stats} -> kept {len(swap_df)}")
    return swap_df[cols]

# (images, captions) of a training batch, for the losses and global retrieval
mscoco_mapper = lambda batch: (batch["image"], batch["caption"])
aro_mapper = lambda batch: (batch["image"], batch["pos_caption"])
svo_mapper = lambda batch: (batch["pos_image"], batch["caption"])

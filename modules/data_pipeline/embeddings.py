import  torch, io, clip, hashlib, zipfile
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

class ImgStream:
    def __init__(self, path, file_type, columns=None):
        if file_type not in ['folder', 'zip', 'df']:
            raise ValueError("file_type must be one of 'folder', 'zip', or 'df'")
        IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg'}

        self.path = path
        self.file_type = file_type
        self.columns = [columns] if isinstance(columns, str) else columns

        if file_type == 'zip':
            with zipfile.ZipFile(path, 'r') as archive:
                self.targets = [n for n in archive.namelist() if Path(n).suffix.lower() in IMG_EXTENSIONS]
            self._length = len(self.targets)
        elif file_type == 'folder':
            self.targets = [p for p in Path(path).rglob('*') if p.suffix.lower() in IMG_EXTENSIONS]
            self._length = len(self.targets)
        elif file_type == 'df':
            if columns is None:
                raise ValueError("For 'df' file_type, 'columns' must be provided.")
            self.df = pd.read_pickle(path)
            self._length = self.df[self.columns].notna().sum().sum()

    def __len__(self):
        return self._length
    
    def __iter__(self):
        if self.file_type == 'zip':
            with zipfile.ZipFile(self.path, 'r') as archive:
                for name in self.targets:
                    yield Path(name).stem, Image.open(io.BytesIO(archive.read(name))).convert("RGB")

        elif self.file_type == 'folder':
            for p in self.targets:
                yield p.stem, Image.open(p).convert("RGB")

        elif self.file_type == 'df':
            for idx, row in self.df.iterrows():
                for col in self.columns:
                    raw_img = row[col]
                    if pd.isna(raw_img) or not raw_img:
                        print(f"Skipping empty or missing image at index {idx}, column '{col}'.")
                        continue
                    img = Image.open(io.BytesIO(raw_img)) if isinstance(raw_img, (bytes, bytearray)) else raw_img
                    img = img.convert("RGB")
                    img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    yield f"{idx}::{col}::{img_hash}", img

def embed_images(data_generator, out_path, df_path = None, device: str = 'cpu'):
    df_store = False
    if df_path and getattr(data_generator, 'file_type', None) == 'df':
        df = pd.read_pickle(df_path)
        for col in data_generator.columns:
            df[f'{col}_id'] = None
        df_store = True
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    embedding_cache = {}

    print("Beginning CLIP embedding extraction...")
    for key, img in tqdm(data_generator):
        try:
            img_t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(img_t).squeeze(0).cpu().float()
            if df_store:
                idx_str, col, img_hash = key.split('::')
                cache_key = int(img_hash, 16)   # md5 hex digest
                df.at[int(idx_str), f'{col}_id'] = cache_key
            else:
                cache_key = int(key)
            embedding_cache[cache_key] = emb
        except Exception as e:
            print(f"Skipping malformed data at key {key}: {e}")
            
    torch.save(embedding_cache, out_path)
    
    if df_store:
        df.to_pickle(df_path)
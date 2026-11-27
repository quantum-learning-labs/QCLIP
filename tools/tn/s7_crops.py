"""Step 7 data: frozen CLIP ViT-B/32 embedding of every PSG object (box expanded to a square around its centre, clipped to the
image, then CLIP's preprocess), in graphs.pt object order -> data/psg/crops_b32.pt (fp16, N_obj x 512). GPU.
  PYTHONPATH=. python tools/tn/s7_crops.py"""
import io, zipfile, torch, clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader

G = torch.load('data/psg/graphs.pt')
_, preprocess = clip.load('ViT-B/32', device='cpu')
class Crops(Dataset):
    def __init__(self): self.z = {}
    def __len__(self): return len(G['file_name'])
    def __getitem__(self, i):
        f = G['file_name'][i]; s = f.split('/')[0]
        if s not in self.z: self.z[s] = zipfile.ZipFile(f'data/mscoco/raw/images/{s}.zip')
        img = Image.open(io.BytesIO(self.z[s].read(f))).convert('RGB'); W, H = img.size
        out = []
        for x1, y1, x2, y2 in G['obj_box'][G['obj_start'][i]:G['obj_start'][i + 1]].tolist():
            cx, cy, h = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 8) / 2
            b = (max(0, cx - h), max(0, cy - h), min(W, cx + h), min(H, cy + h))
            out.append(preprocess(img.crop(b)))
        return torch.stack(out) if out else torch.zeros(0, 3, 224, 224)
dev = 'cuda'
model, _ = clip.load('ViT-B/32', device=dev); model.eval()
feats = []
loader = DataLoader(Crops(), batch_size=16, num_workers=8, collate_fn=lambda b: torch.cat(b))
with torch.no_grad():
    for k, x in enumerate(loader):
        if len(x): feats.append(model.encode_image(x.to(dev, non_blocking=True)).half().cpu())
        if k % 200 == 0: print(k * 16, 'images', flush=True)
feats = torch.cat(feats)
assert len(feats) == int(G['obj_start'][-1]), (len(feats), int(G['obj_start'][-1]))
torch.save(feats, 'data/psg/crops_b32.pt'); print('crops', tuple(feats.shape))

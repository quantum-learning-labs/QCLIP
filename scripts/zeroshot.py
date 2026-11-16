# Zero-shot references (no training): pretrained vision-language models scored on the test sets of each split.
#   python scripts/zeroshot.py --dataset svo-probes --images data/svo-probes/raw/images.zip --seeds 0 1 2 3 4 -o results/v1/zeroshot
#   python scripts/zeroshot.py --dataset aro --images data/aro/raw/crops.zip ...
# Rows are exactly those of the trained models' test sets (configs/data/<dataset>.yaml, dataset.split.seed = seed), with
# raw images (ARO: the row's box cropped, as in the official evaluation). Scores: CLIP / OpenCLIP cosine of the image and
# text embeddings; BLIP the image-text matching (ITM) probability and, separately, the contrastive (ITC) cosine.
# Accuracy as in the evaluator: hard_neg_acc (pos > neg) and hard_neg_acc_tie_half (a tie counts 1/2).
import argparse, io, json, os, time, zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from modules.config import setup_exp, git_state
from modules.data_pipeline.engine import DataEngine

MODELS = {   # name -> (HF model id, kind)
    'clip': ('openai/clip-vit-base-patch32', 'clip'),
    'openclip': ('laion/CLIP-ViT-B-32-laion2B-s34B-b79K', 'clip'),
    'blip': ('Salesforce/blip-itm-base-coco', 'blip'),
}
# (image column, positive text column, negative ...) per task: image choice has one caption and two images
LAYOUT = {'evaluate_image_choice': ('caption', 'corrected_sentence', 'pos_image_id', 'neg_image_id'),
          'evaluate_text_choice_svo': ('image', 'pos_image_id', 'corrected_sentence', 'swapped_sentence'),
          'evaluate_text_choice_aro': ('image', 'image_id', 'true_caption', 'false_caption')}

class Images:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        names = self.zip.namelist()
        self.parent, self.suffix = names[0], Path(names[1]).suffix
    def __call__(self, image_id):
        return Image.open(io.BytesIO(self.zip.read(f"{self.parent}{int(image_id)}{self.suffix}"))).convert("RGB")

class Scorer:
    def __init__(self, name, device):
        from transformers import AutoProcessor
        self.model_id, self.kind = MODELS[name]
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if self.kind == 'clip':
            from transformers import CLIPModel
            self.model = CLIPModel.from_pretrained(self.model_id).to(device).eval()
        else:
            from transformers import BlipForImageTextRetrieval
            self.model = BlipForImageTextRetrieval.from_pretrained(self.model_id).to(device).eval()
        self.revision = getattr(self.model.config, '_commit_hash', None)

    @torch.no_grad()
    def scores(self, images, texts):
        # one score per (image, text) pair, for each head: {'cos': ...} or {'itm': ..., 'itc': ...}
        inputs = self.processor(images=images, text=texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        if self.kind == 'clip':
            out = self.model(**inputs)
            i = torch.nn.functional.normalize(out.image_embeds, dim=-1)
            t = torch.nn.functional.normalize(out.text_embeds, dim=-1)
            return {'cos': (i * t).sum(-1).float().cpu()}
        itm = self.model(**inputs, use_itm_head=True).itm_score.softmax(-1)[:, 1]
        itc = self.model(**inputs, use_itm_head=False).itm_score.diagonal()   # ITC: image x text cosine matrix, pairs on the diagonal
        return {'itm': itm.float().cpu(), 'itc': itc.float().cpu()}

def pair_scores(scorer, images, rows, layout, batch, cache):
    # score of every row's (image, positive) and (image, negative) pair, cached by (image id, text)
    kind, a, b, c = layout
    pairs = []
    for r in rows:
        if kind == 'caption':
            pairs += [(r[b], r[a]), (r[c], r[a])]
        else:
            pairs += [(r[a], r[b]), (r[a], r[c])]
    todo = [p for p in dict.fromkeys(pairs) if p not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        out = scorer.scores([images(p[0]) for p in chunk], [str(p[1]) for p in chunk])
        for j, p in enumerate(chunk):
            cache[p] = {h: float(v[j]) for h, v in out.items()}
    return [(cache[pairs[2 * k]], cache[pairs[2 * k + 1]]) for k in range(len(rows))]

def accuracy(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    tie = np.isclose(pos, neg, rtol=1e-5, atol=1e-8)   # as the evaluator
    return {'hard_neg_acc': float((pos > neg).mean()), 'hard_neg_acc_tie_half': float(((pos > neg) & ~tie).mean() + 0.5 * tie.mean()),
            'ties': int(tie.sum()), 'n': int(len(pos))}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Zero-shot CLIP / OpenCLIP / BLIP on the test sets of each split")
    ap.add_argument('--dataset', required=True, choices=['svo-probes', 'aro'])
    ap.add_argument('--images', required=True, help='zip of images/<image id>.jpg (SVO: raw/images.zip, ARO: raw/crops.zip)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--models', nargs='+', default=list(MODELS))
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--limit', type=int, default=None, help='first N rows of each test set (smoke test)')
    ap.add_argument('-o', '--out', required=True, help='output directory: <model>_k<seed>.json')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = Images(args.images)
    for name in args.models:
        scorer, cache = Scorer(name, device), {}
        for seed in args.seeds:
            path = Path(args.out) / f"{name}_k{seed}.json"
            if path.exists():
                print(f"skip {path} (exists)"); continue
            config, _, _ = setup_exp('configs/vqc.yaml', overrides=[f'dataset={args.dataset}', f'dataset.split.seed={seed}'])
            engine = DataEngine(config, None, device)
            results, t0 = {}, time.time()
            for test_name, entry in config['dataset']['test'].items():
                rows = engine.load_text(test_name)
                if args.limit:
                    rows = rows.iloc[:args.limit]
                task = entry['task'] if entry['task'] == 'evaluate_image_choice' else \
                    ('evaluate_text_choice_aro' if args.dataset == 'aro' else 'evaluate_text_choice_svo')
                pairs = pair_scores(scorer, images, rows.to_dict('records'), LAYOUT[task], args.batch, cache)
                for head in pairs[0][0]:
                    acc = accuracy([p[head] for p, _ in pairs], [n[head] for _, n in pairs])
                    results.update({f"{test_name}_{head}_{k}": v for k, v in acc.items()})
            record = engine.split_record()
            json.dump({'model': name, 'model_id': scorer.model_id, 'hf_revision': scorer.revision, 'dataset': args.dataset,
                       'split_seed': seed, 'split': record, 'limit': args.limit, 'git_commit': git_state(),
                       'device': device, 'seconds': round(time.time() - t0, 1), 'results': results}, open(path, 'w'), indent=1)
            print(name, seed, {k: round(v, 4) for k, v in results.items() if k.endswith('tie_half')})

# Classical baseline for the backend benchmark: CLIP ViT-B/32 zero-shot on the SVO-probes swap sets (same image
# embeddings and caption pairs as the quantum circuits: image pos_image_id, corrected vs swapped sentence), with GPU
# energy per decision from the NVML energy counter (and host CPU package energy from RAPL when readable).
# A decision = encode 2 captions + 2 cosines (the image embedding is precomputed, as on the quantum side); the image
# encoder is timed separately (front-end cost shared by every model).
#   PYTHONPATH=. python analysis/bench/classical_energy.py -o results/bench_b2/classical
import argparse, glob, json, pickle, time
from pathlib import Path
import numpy as np
import torch, clip, pynvml


def rapl_uj():
    fs = sorted(glob.glob('/sys/class/powercap/intel-rapl:[0-9]/energy_uj'))
    try:
        return sum(int(open(f).read()) for f in fs) if fs else None
    except OSError:
        return None


def measure(fn, n_items, min_s, h):
    # run fn() until min_s seconds have passed; energy per item, total and above idle
    torch.cuda.synchronize()
    e0, c0, t0, reps = pynvml.nvmlDeviceGetTotalEnergyConsumption(h), rapl_uj(), time.time(), 0
    while time.time() - t0 < min_s:
        fn(); reps += 1
        torch.cuda.synchronize()
    dt, e = time.time() - t0, (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1e3
    c1 = rapl_uj()
    n = reps * n_items
    out = {'items': n, 'seconds': dt, 'gpu_J': e, 'gpu_J_per_item': e / n, 's_per_item': dt / n, 'gpu_W': e / dt}
    if c0 is not None and c1 is not None and c1 >= c0:
        out['cpu_pkg_J_per_item'] = (c1 - c0) / 1e6 / n
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--out', required=True)
    p.add_argument('--min_s', type=float, default=30)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    dev = 'cuda'
    pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(0)
    model, _ = clip.load('ViT-B/32', device=dev); model.eval()
    emb = torch.load('data/svo-probes/processed/image_embeddings.pt', map_location='cpu')
    res = {'gpu': pynvml.nvmlDeviceGetName(h), 'model': 'CLIP ViT-B/32 zero-shot', 'rapl': rapl_uj() is not None}

    # idle GPU power (the card's own floor, counted in every measurement below)
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h); time.sleep(10)
    res['idle_W'] = (pynvml.nvmlDeviceGetTotalEnergyConsumption(h) - e0) / 1e3 / 10

    for name in ['swap', 'swap_full', 'swap_animate']:
        df = pickle.load(open(f'data/svo-probes/processed/{name}.pkl', 'rb'))
        img = torch.stack([torch.as_tensor(emb[int(i)]).float().flatten() for i in df['pos_image_id']]).to(dev)
        img = img / img.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            tp = model.encode_text(clip.tokenize(list(df['corrected_sentence'])).to(dev)).float()
            tn = model.encode_text(clip.tokenize(list(df['swapped_sentence'])).to(dev)).float()
        tp, tn = tp / tp.norm(dim=-1, keepdim=True), tn / tn.norm(dim=-1, keepdim=True)
        sp, sn = (img * tp).sum(-1), (img * tn).sum(-1)
        res[f'acc_{name}'] = float(((sp > sn).float() + 0.5 * (sp == sn).float()).mean())
        res[f'pairs_{name}'] = len(df)
        if name != 'swap':
            continue
        toks = torch.cat([clip.tokenize(list(df['corrected_sentence'])), clip.tokenize(list(df['swapped_sentence']))]).to(dev)
        n = len(df)

        def batch():   # all 95 decisions at once
            with torch.no_grad():
                t = model.encode_text(toks).float(); t = t / t.norm(dim=-1, keepdim=True)
                ((img * t[:n]).sum(-1) > (img * t[n:]).sum(-1)).sum().item()

        def single():  # one decision at a time (batch 2)
            with torch.no_grad():
                for i in range(n):
                    t = model.encode_text(toks[[i, n + i]]).float(); t = t / t.norm(dim=-1, keepdim=True)
                    ((img[i] * t[0]).sum() > (img[i] * t[1]).sum()).item()

        for f in (batch, single):
            f()
        res['decision_batched'] = measure(batch, n, a.min_s, h)
        res['decision_single'] = measure(single, n, a.min_s, h)
        x = torch.randn(64, 3, 224, 224, device=dev, dtype=next(model.parameters()).dtype)

        def images():
            with torch.no_grad():
                model.encode_image(x).sum().item()
        images()
        res['image_encode_batch64'] = measure(images, 64, a.min_s, h)
        print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(out / 'classical.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

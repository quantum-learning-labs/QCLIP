"""Step 4, phase 0a: parameter norms of TN checkpoints (CPU).

Per checkpoint: word-vector norm quantiles; per type tensor / MPS core: shape, Frobenius norm, RMS*sqrt(128) (= 1 at
norm init for a 128-dim output leg) and the top singular value of its unfolding along each leg (the largest gain of the
multilinear map on unit inputs, for whichever leg is the output). Usage: s4_norms.py out.json ckpt.pt [ckpt.pt ...]
"""
import json, math, sys
import torch

def stats(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    text, out = sd['text'], {'epoch': sd.get('epoch'), 'types': {}}
    words = torch.stack([v.float() for k, v in text.items() if k.startswith('params.w:')])
    wn = words.norm(dim=1)
    out['words'] = {'n': len(wn), **{f'q{q}': float(wn.quantile(q / 100)) for q in (0, 1, 50, 99, 100)}}
    for k, v in text.items():
        if not k.startswith('params.') or k.startswith('params.w:'):
            continue
        v = v.float()
        legs = [float(torch.linalg.matrix_norm(v.movedim(i, 0).reshape(v.shape[i], -1), ord=2)) for i in range(v.dim())] if v.dim() >= 2 else []
        out['types'][k[len('params.'):]] = {'shape': list(v.shape), 'fro': float(v.norm()),
                                           'rms128': float(v.pow(2).mean().sqrt()) * math.sqrt(128), 'sigma_legs': legs}
    out['norm_head'] = {k: float(v.norm()) for k, v in text.items() if k.startswith('norm_head')}
    return out

if __name__ == '__main__':
    res = {p: stats(p) for p in sys.argv[2:]}
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
    for p, r in res.items():
        t = r['types'].values()
        smax = sorted((max(x['sigma_legs']) for x in t if x['sigma_legs']), reverse=True)
        rms = sorted(x['rms128'] for x in t)
        print(f"{p.split('/')[-2]} ep {r['epoch']}: words |w| q0/50/100 {r['words']['q0']:.2f}/{r['words']['q50']:.2f}/{r['words']['q100']:.2f}"
              f" | {len(rms)} type tensors rms128 min/med/max {rms[0]:.2f}/{rms[len(rms)//2]:.2f}/{rms[-1]:.2f}"
              f" | top sigma_max {', '.join(f'{s:.1f}' for s in smax[:5])} med {smax[len(smax)//2]:.2f} | head {r['norm_head']}")

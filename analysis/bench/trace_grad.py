# VQCModel discard trace_mode 'psi' vs 'auto': rho, gradients, forward+backward time and peak GPU memory on real sentences.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/trace_grad.py -cp <discard ckpt>... [--n 256] [--bs 64] [--modes psi,auto]
import argparse, json, time
from pathlib import Path
import pandas as pd, torch
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config
from modules.data_pipeline.engine import DataEngine
from trace_check import recipes_of


def run(ckpt, mode, n, bs, dev):
    config, _, _ = setup_exp(resolve_eval_config(None, ckpt), seed=0)
    config['text']['trace_mode'] = mode
    ansatz, image_model, text_model, _ = build_experiment(config, dev)
    de = DataEngine(config, ansatz, dev)
    c_val = de.compile_text('val')
    de.text_init(text_model, pd.concat([de.compile_text('train'), c_val], ignore_index=True))
    CheckpointManager.load_model(ckpt, image_model, text_model, dev); text_model.to(dev)
    rec = recipes_of(c_val)[:n]
    g = torch.Generator().manual_seed(0)
    d = 2 ** text_model.out_q
    W = torch.randn(d, d, generator=g, dtype=torch.complex64).to(dev)
    if dev == 'cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    text_model.zero_grad(); rhos = []; t0 = time.time()
    for i in range(0, len(rec), bs):
        rho = text_model(rec[i:i + bs])
        (rho * W).real.sum().backward(); rhos.append(rho.detach().cpu())
    if dev == 'cuda': torch.cuda.synchronize()
    t = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == 'cuda' else None
    grad = torch.cat([p.grad.detach().flatten().cpu() for p in text_model.params if p.grad is not None])
    n_dbl = sum(v.get('doubled', False) for v in text_model.plan_cache.values())
    return torch.cat(rhos), grad, t, peak, n_dbl, len(text_model.plan_cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('--n', type=int, default=256); ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--modes', default='psi,auto'); ap.add_argument('-o', '--out', default='results/shotcost/trace_grad.json')
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = {}
    for c in a.checkpoints:
        ck = Path(c); r = {}; out = {}
        for mode in a.modes.split(','):
            try:
                out[mode] = run(ck, mode, a.n, a.bs, dev)
                rho, grad, t, peak, nd, npl = out[mode]
                r[mode] = dict(seconds=t, peak_GiB=peak, doubled_structures=nd, structures=npl,
                               zero_trace=int((rho.diagonal(dim1=1, dim2=2).sum(-1).real.abs() < 1e-6).sum()))
            except torch.OutOfMemoryError as e:
                r[mode] = {'oom': str(e)[:120]}; torch.cuda.empty_cache()
        if 'psi' in out and 'auto' in out:
            (r1, g1, *_), (r2, g2, *_) = out['psi'], out['auto']
            r['rho_max_abs'] = float((r1 - r2).abs().max())
            r['grad_max_rel'] = float((g1 - g2).abs().max() / g1.abs().max())
            r['grad_norm_rel'] = float((g1 - g2).norm() / g1.norm())
        res[ck.parent.name] = r
        print(ck.parent.name, dev, json.dumps(r), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

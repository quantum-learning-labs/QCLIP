# Gradient statistics of the two readouts vs the number m of non-sentence wires (cluster GPU job).
# For random parameter draws (training init: small angles near identity, and uniform angles), per sentence:
#   cost_tr = <phi|rho|phi>  (partial trace),  cost_ps = |<phi|psi_0>|^2 (post-selected, renormalised), p_0.
# We record E[|grad|^2] per parameter that the sentence uses, grouped by m.  usage: grad_probe.py <out.json> <config.yaml>
import sys, json, yaml, collections, torch, numpy as np, pandas as pd
sys.argv += [] ; out, cfg_path = sys.argv[1], sys.argv[2]
from modules.utils.factory import build_experiment
from modules.utils.general import set_seed
from modules.data_pipeline.engine import DataEngine
import os; exec(open(os.path.expanduser('~/qclip_hpc/gap/readout_swap_fns.py')).read())   # psi_forward
DEV = torch.device('cuda')
cfg = yaml.safe_load(open(cfg_path)); cfg['text']['discard'] = True; cfg['loss'] = 'hs'; cfg['seed'] = 0
res = collections.defaultdict(lambda: collections.defaultdict(list))
for layers in (1, 2, 3):
    cfg['text']['layers'] = layers; set_seed(0)
    ans, img, txt, _ = build_experiment(cfg, DEV)
    de = DataEngine(cfg, ans, DEV); cva = de.compile_text('val'); ctr = de.compile_text('train')
    de.text_init(txt, pd.concat([ctr, cva], ignore_index=True))
    txt.to(DEV)
    recipes = list(zip(cva['corrected_sentence_einsum'], cva['corrected_sentence_symbols']))
    ms = [len(e.split('->')[1]) - cfg['embedding_qubits'] for e, _ in recipes]
    by_m = collections.defaultdict(list)
    for r, m in zip(recipes, ms): by_m[m].append(r)
    for init in ('train_init', 'uniform'):
        for draw in range(12):
            g = torch.Generator().manual_seed(1000 + draw)
            with torch.no_grad():
                for p in txt.params:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.01 if init == 'train_init' else torch.atanh(torch.rand(p.shape, generator=g) * 1.98 - 0.99))
            phi = torch.randn(2 ** cfg['embedding_qubits'], dtype=torch.complex64, generator=g).to(DEV); phi = phi / phi.norm()
            for m, rs in by_m.items():
                rs = rs[:16]
                for name in ('tr', 'ps'):
                    txt.zero_grad()
                    psis = psi_forward(txt, rs)
                    cost = 0
                    for p in psis:
                        if name == 'tr': cost = cost + (phi.conj() @ (p @ p.conj().T) @ phi).real
                        else:
                            psi0 = p[:, 0] / p[:, 0].norm().clamp(min=1e-12); cost = cost + (phi.conj() @ psi0).abs() ** 2
                    (cost / len(psis)).backward()
                    gr = torch.cat([q.grad.reshape(-1) for q in txt.params if q.grad is not None])
                    used = gr[gr != 0]
                    res[f'L{layers}_{init}_{name}'][m].append(float((used ** 2).mean()) if len(used) else 0.0)
                with torch.no_grad():
                    res[f'L{layers}_{init}_p0'][m].append(float(np.mean([float((p[:, 0].abs() ** 2).sum()) for p in psi_forward(txt, rs)])))
            print(layers, init, draw, flush=True)
summary = {k: {str(m): [float(np.mean(v)), float(np.std(v)), len(v)] for m, v in sorted(d.items())} for k, d in res.items()}
json.dump(summary, open(out, 'x'), indent=1)
print(json.dumps(summary, indent=1))

# Discard text states: "build |psi> over all wires, then rho = psi psi^dag" (current VQCModel.forward) vs a doubled
# ket/bra network in which each discarded wire is traced inside the contraction (never materialising 2^nq).
# (1) equality of rho on real sentences, (2) largest intermediate / flops of both contraction paths per sentence structure.
# usage: PYTHONPATH=. python analysis/bench/trace_check.py -cp <discard ckpt>... [--n_eq 256] [-o out.json]
import argparse, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
import opt_einsum as oe
from modules.utils.factory import build_experiment
from modules.utils.general import setup_exp, CheckpointManager, resolve_eval_config
from modules.data_pipeline.engine import DataEngine


def doubled_str(einsum_str, out_q):
    inp, out = einsum_str.split('->')
    terms = inp.split(',')
    sent, traced = out[:out_q], set(out[out_q:])
    used = set(einsum_str) - set(',->')
    fresh, k, m = [], 0, {}
    for c in sorted(used - traced):
        while oe.get_symbol(k) in used or oe.get_symbol(k) == '$' or oe.get_symbol(k) in fresh: k += 1
        m[c] = oe.get_symbol(k); fresh.append(m[c]); k += 1
    bra = [''.join(c if c in traced else m[c] for c in t) for t in terms]
    return ','.join('$' + t for t in terms + bra) + '->$' + sent + ''.join(m[c] for c in sent)


def operands(model, einsum_str, tensor_lists):
    # same operand construction as VQCModel.forward
    thetas = torch.tanh(torch.cat([p for p in model.params])) * 2 * torch.pi
    plan = model._plan(einsum_str, tensor_lists[0]); B = len(tensor_lists)
    ops = [None] * plan['n_ops']
    for pos, buf_name in plan['static']:
        buf = getattr(model, buf_name); ops[pos] = buf.expand(B, *buf.shape)
    if plan['param_pos']:
        idx = torch.tensor([[model.sym2param.get(t[p]['name'], model.unk_param_index) for p in plan['param_pos']] for t in tensor_lists])
        angles = thetas[idx].to(dtype=model.precision)
        for op_type, cols in plan['by_type'].items():
            g = model.GATE_FNS[op_type](angles[:, cols])
            for j, c in enumerate(cols): ops[plan['param_pos'][c]] = g[:, j]
    return ops


def rho_doubled(model, recipes):
    groups = defaultdict(list)
    for i, (s, t) in enumerate(recipes): groups[s].append((i, t))
    d = 2 ** model.out_q
    out = torch.zeros(len(recipes), d, d, dtype=model.precision)
    for s, items in groups.items():
        idx, tl = zip(*items)
        ops = operands(model, s, tl)
        r = oe.contract(doubled_str(s, model.out_q), *ops, *[o.conj() for o in ops], backend='torch').reshape(len(idx), d, d)
        if model.renorm: r = r / r.diagonal(dim1=1, dim2=2).sum(-1).real.clamp(min=1e-12)[:, None, None]
        out[list(idx)] = r
    return out


def recipes_of(df):
    # (einsum, symbols) recipes from every *_einsum / *_symbols column pair of a compiled split (list cells expanded)
    out = []
    for col in [c for c in df.columns if c.endswith('_einsum')]:
        sc = col[:-len('_einsum')] + '_symbols'
        if sc not in df.columns: continue
        for e, t in zip(df[col], df[sc]):
            if isinstance(e, float): continue
            if isinstance(e, (list, tuple)) and e and isinstance(e[0], str) and '->' in e[0]: out += list(zip(e, t))
            else: out.append((e, t))
    return out


def path_stats(einsum_str, tensors, out_q, B=64):
    inp, out = einsum_str.split('->')
    shp = [(B, *[2] * len(t)) for t in inp.split(',')]
    cur = oe.contract_path(f"{','.join('$' + t for t in inp.split(','))}->${out}", *shp, shapes=True)[1]
    dbl = oe.contract_path(doubled_str(einsum_str, out_q), *(shp * 2), shapes=True)[1]
    return dict(nq=len(out), cur_largest=int(cur.largest_intermediate), dbl_largest=int(dbl.largest_intermediate),
                cur_flops=float(cur.opt_cost), dbl_flops=float(dbl.opt_cost))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('--n_eq', type=int, default=256)
    ap.add_argument('-o', '--out', default='results/shotcost/trace_check.json')
    a = ap.parse_args()
    res = {}
    for c in a.checkpoints:
        ckpt = Path(c); t0 = time.time()
        config, DEV, _ = setup_exp(resolve_eval_config(None, ckpt), seed=0)
        assert config['text'].get('discard'), 'needs a discard checkpoint'
        ansatz, image_model, text_model, _ = build_experiment(config, DEV)
        de = DataEngine(config, ansatz, DEV)
        c_val = de.compile_text('val')
        de.text_init(text_model, pd.concat([de.compile_text('train'), c_val], ignore_index=True))
        CheckpointManager.load_model(ckpt, image_model, text_model, DEV); text_model.to('cpu'); text_model.eval()
        r = {'e': config['embedding_qubits'], 'n': config['text'].get('n'), 'out_q': text_model.out_q}
        # (1) equality on real sentences (float64 copy of the model for a clean comparison as well)
        recipes = recipes_of(c_val)[:a.n_eq]
        with torch.no_grad():
            ref = text_model(recipes); new = rho_doubled(text_model, recipes)
        diff = (ref - new).abs().amax(dim=(1, 2))
        r.update(eq_n=len(recipes), eq_max_abs=float(diff.max()), eq_median_abs=float(diff.median()),
                 ref_trace_min=float(ref.diagonal(dim1=1, dim2=2).sum(-1).real.min()))
        # (2) path statistics per distinct sentence structure (val + test std)
        stats = []
        seen = set()
        for comp in (c_val, de.compile_text('std')):
            for s, t in recipes_of(comp):
                if s in seen: continue
                seen.add(s); stats.append(path_stats(s, t, text_model.out_q))
        st = pd.DataFrame(stats)
        r.update(structures=len(st), nq_max=int(st.nq.max()),
                 cur_largest_max=int(st.cur_largest.max()), dbl_largest_max=int(st.dbl_largest.max()),
                 ratio_largest_median=float((st.dbl_largest / st.cur_largest).median()),
                 cur_flops_sum=float(st.cur_flops.sum()), dbl_flops_sum=float(st.dbl_flops.sum()),
                 by_nq=st.groupby('nq')[['cur_largest', 'dbl_largest']].max().reset_index().to_dict('records'), wall_s=time.time() - t0)
        res[ckpt.parent.name] = r
        print(ckpt.parent.name, {k: v for k, v in r.items() if k != 'by_nq'}, flush=True)
        print('  largest intermediate by nq (cur, doubled), batch 64:', [(x['nq'], x['cur_largest'], x['dbl_largest']) for x in r['by_nq']], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()

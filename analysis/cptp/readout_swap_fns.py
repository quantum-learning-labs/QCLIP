import collections, torch
def psi_forward(tm, recipes):
    groups = collections.defaultdict(list)
    for i, (e, t) in enumerate(recipes): groups[e].append((i, t))
    thetas = torch.tanh(torch.cat([p for p in tm.params])) * 2 * torch.pi
    out = [None] * len(recipes)
    for e, items in groups.items():
        idx, tl = zip(*items); mb = len(idx)
        plan = tm._plan(e, tl[0]); ops = [None] * plan['n_ops']
        for pos, b in plan['static']:
            buf = getattr(tm, b); ops[pos] = buf.expand(mb, *buf.shape)
        if plan['param_pos']:
            pidx = torch.tensor([[tm.sym2param.get(ts[p]['name'], tm.unk_param_index) for p in plan['param_pos']] for ts in tl], device=thetas.device)
            ang = thetas[pidx].to(tm.precision)
            for op, cols in plan['by_type'].items():
                g = tm.GATE_FNS[op](ang[:, cols])
                for j, c in enumerate(cols): ops[plan['param_pos'][c]] = g[:, j]
        go = plan['expr'](*ops, backend='torch')
        assert go.numel() % (mb * 2 ** tm.out_q) == 0, e
        go = go.reshape(mb, 2 ** tm.out_q, -1)
        for j, i in enumerate(idx): out[i] = go[j]
    return out


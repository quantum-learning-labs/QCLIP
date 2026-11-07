# E3: how good are our contraction paths, and what is the least any exact contraction can cost?
# For the first --k typical decisions of each checkpoint, on the post-selection network (phi0) and the discard network (rho):
#   upper bounds: log2 largest intermediate ("width") and flops of the opt_einsum paths greedy / random-greedy / auto-hq
#   lower bound : MMD+ (min-d) treewidth lower bound of the network's line graph (indices = vertices, joined when they share a tensor;
#                 open output indices joined into one clique). Markov & Shi (2008): the best pairwise contraction width equals the line-graph
#                 treewidth, so any exact pairwise contraction (statevector included) must hold an intermediate of ~2^lb entries.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/tw_bounds.py -o out.csv -cp ckpt... [--k 5] [--opts greedy,random-greedy-128,ctg]
import argparse, csv, time
from pathlib import Path
import numpy as np, opt_einsum as oe, torch, networkx as nx
from inf_engines import TN, load


def line_graph(eq):
    ins, out = eq.split('->'); G = nx.Graph()
    for term in ins.split(',') + [out]:
        idx = list(term); G.add_nodes_from(idx)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)): G.add_edge(idx[i], idx[j])
    return G


def mmd_plus(G):
    """MMD+ (min-d) treewidth lower bound: repeatedly take a min-degree vertex, record its degree, contract it into its min-degree neighbour."""
    G = G.copy(); lb = 0
    adj = {v: set(G.neighbors(v)) for v in G.nodes}
    while len(adj) > 1:
        v = min(adj, key=lambda x: len(adj[x])); d = len(adj[v]); lb = max(lb, d)
        nb = adj.pop(v)
        if not nb: continue
        u = min(nb, key=lambda x: len(adj[x]))
        for w in nb:
            adj[w].discard(v)
            if w != u: adj[w].add(u); adj[u].add(w)
    return lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--opts', default='greedy,random-greedy-128,ctg')
    ap.add_argument('--pairs', type=int, default=40)
    a = ap.parse_args()
    out = Path(a.out); new = not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(open(out, 'a', newline=''))
    if new: w.writerow(['ckpt', 'text_n', 'layers', 'nq', 'idx', 'net', 'n_tensors', 'n_indices', 'tw_lb', 'opt', 'width', 'log10_flops', 'path_s'])
    for c in a.checkpoints:
        ckpt = Path(c); config, nq, D = load(ckpt, a); L = config['text'].get('layers')
        for i, (qt, _, _, oq) in enumerate(D[:a.k]):
            for net, rho in (('ps', False), ('rho', True)):
                T = TN(qt, oq, 'cpu', torch.complex64, rho=rho, opt='greedy')
                arrs = [x.numpy() for x in T.arrs]
                G = line_graph(T.eq); t0 = time.time(); lb = mmd_plus(G)
                print(f'{ckpt.parent.name} #{i} {net}: {len(arrs)} tensors, {G.number_of_nodes()} indices, tw lb {lb} ({time.time() - t0:.1f}s)', flush=True)
                for opt in a.opts.split(','):
                    t0 = time.time()
                    try:
                        o = opt
                        if opt == 'ctg':                       # cotengra hyper-optimiser, 30 s budget: only as a check of path quality
                            import cotengra as ctg
                            o = ctg.HyperOptimizer(max_time=30, max_repeats=10 ** 6, minimize='flops', progbar=False)
                        _, info = oe.contract_path(T.eq, *arrs, optimize=o)
                        wd, fl = float(np.log2(max(float(info.largest_intermediate), 1.0))), float(np.log10(max(float(info.opt_cost), 1.0)))
                    except Exception as e:
                        print('   ', opt, 'FAILED', type(e).__name__, str(e)[:120], flush=True); wd = fl = float('nan')
                    w.writerow([ckpt.parent.name, config['text'].get('n'), L, nq, i, net, len(arrs), G.number_of_nodes(), lb, opt, wd, fl, time.time() - t0])
                    print(f'    {opt}: width {wd:.0f} log10 flops {fl:.2f} ({time.time() - t0:.1f}s)', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()

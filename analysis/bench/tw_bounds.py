# E3: how good are our contraction paths, and what is the least any exact contraction can cost?
# For the first --k typical decisions of each checkpoint, on the post-selection network (phi0) and the discard network (rho):
#   upper bounds: log2 largest intermediate ("width") and flops of the opt_einsum paths greedy / random-greedy / auto-hq
#   lower bound : MMD+ (min-d) treewidth lower bound of the network's line graph (indices = vertices, joined when they share a tensor;
#                 open output indices joined into one clique). Markov & Shi (2008): the best pairwise contraction width equals the line-graph
#                 treewidth, so any exact pairwise contraction (statevector included) must hold an intermediate of ~2^lb entries.
# usage: PYTHONPATH=.:analysis/bench python analysis/bench/tw_bounds.py -o out.csv -cp ckpt... [--k 5] [--opts greedy,random-greedy-128,ctg]
import argparse, csv, signal, time
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


def _alarm(*_): raise TimeoutError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-cp', '--checkpoints', nargs='+', required=True)
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--opts', default='greedy,random-greedy-128,ctg')
    ap.add_argument('--pairs', type=int, default=40)
    ap.add_argument('--ctg_s', type=float, default=30, help='cotengra time budget per network (s)')
    ap.add_argument('--ctg_methods', default='greedy', help='cotengra path methods (no kahypar in the envs); labels trials on 8-layer networks gave'
                    ' 10^94-flop trees whose reconfiguration ran for hours past the hard stop')
    ap.add_argument('--ctg_reconf', action='store_true', help='cotengra subtree reconfiguration of each trial')
    ap.add_argument('--nets', default='ps,rho')
    a = ap.parse_args()
    out = Path(a.out); new = not out.exists(); out.parent.mkdir(parents=True, exist_ok=True)
    w = csv.writer(open(out, 'a', newline='', buffering=1))
    if new: w.writerow(['ckpt', 'text_n', 'layers', 'nq', 'idx', 'net', 'n_tensors', 'n_indices', 'tw_lb', 'opt', 'width', 'log10_flops', 'path_s'])
    for c in a.checkpoints:
        ckpt = Path(c); config, nq, D = load(ckpt, a); L = config['text'].get('layers')
        for i, (qt, _, _, oq) in enumerate(D[:a.k]):
            for net, rho in [t for t in (('ps', False), ('rho', True)) if t[0] in a.nets.split(',')]:
                T = TN(qt, oq, 'cpu', torch.complex64, rho=rho, opt='greedy')
                arrs = [x.numpy() for x in T.arrs]
                G = line_graph(T.eq); t0 = time.time(); lb = mmd_plus(G)
                print(f'{ckpt.parent.name} #{i} {net}: {len(arrs)} tensors, {G.number_of_nodes()} indices, tw lb {lb} ({time.time() - t0:.1f}s)', flush=True)
                for opt in a.opts.split(','):
                    t0 = time.time()
                    try:
                        o = opt
                        if opt == 'ctg':                       # cotengra hyper-optimiser (--ctg_s budget, optional subtree reconfiguration)
                            import cotengra as ctg
                            o = ctg.HyperOptimizer(methods=a.ctg_methods.split(','), max_time=a.ctg_s, max_repeats=10 ** 6, minimize='flops',
                                                   reconf_opts={} if a.ctg_reconf else None, parallel=False, progbar=False)
                        if opt == 'ctg':                       # hard stop: one trial (or its reconfiguration) can overrun max_time by hours on 8-layer networks;
                            signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(a.ctg_s * 1.5) + 1)   # then take the best tree found so far
                            try: oe.contract_path(T.eq, *arrs, optimize=o)
                            except TimeoutError: print('    ctg: hard stop, best tree so far', flush=True)
                            finally: signal.alarm(0)
                            o = o.tree.get_path()
                        _, info = oe.contract_path(T.eq, *arrs, optimize=o)
                        wd, fl = float(np.log2(max(float(info.largest_intermediate), 1.0))), float(np.log10(max(float(info.opt_cost), 1.0)))
                    except Exception as e:
                        print('   ', opt, 'FAILED', type(e).__name__, str(e)[:120], flush=True); wd = fl = float('nan')
                    w.writerow([ckpt.parent.name, config['text'].get('n'), L, nq, i, net, len(arrs), G.number_of_nodes(), lb, opt, wd, fl, time.time() - t0])
                    print(f'    {opt}: width {wd:.0f} log10 flops {fl:.2f} ({time.time() - t0:.1f}s)', flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main()

"""Treewidth of the caption networks with and without Frobenius lifting (step 5; viva M2 / H: cost must come from the graph).
Graph: one vertex per word box (type tensor + word vector merged) and per extra map (m:...), one hyperedge per wire after the
spider merges (a spider = a shared index = a clique on its boxes); open output wire dropped. Reports the min-fill upper bound
on treewidth (box graph, and index graph = one vertex per wire, a clique per box: contraction cost is exponential in
its treewidth, Markov & Shi 2008), the cyclomatic number of the box graph, and the largest spider (hyperedge size), over N training captions.
  PYTHONPATH=. python tools/tn/s5_treewidth.py <out.json> [N=20000]"""
import sys, json, collections, itertools
from itertools import count
from pathlib import Path
import pandas as pd, networkx as nx
from networkx.algorithms.approximation import treewidth_min_fill_in
from modules.compilation.classical.lift import RULES, load_rootmap, lift

out_path = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
if out_path.exists(): raise FileExistsError(out_path)
rm = load_rootmap('modules/compilation/classical/e2_rootmap.json')
caps = [tn for cs in pd.read_pickle('data/mscoco/processed/tn/train.pkl')['captions_diagram'] for tn in (cs or [])][:N]
res = {}
_last = [0]
def lw_last(c): return _last[0]
for name, rules in (('none', None), ('R1', ('R1',)), ('frobenius', tuple(RULES))):
    tw, cyc, hyp, n_box, lw, lw_len = collections.Counter(), collections.Counter(), collections.Counter(), [], collections.Counter(), collections.defaultdict(list)
    for tn in caps:
        if rules:
            items, extras, uf, _ = lift(tn, rules, rm, count(max(i for _, ix, _ in tn for i in ix) + 1))
            boxes = [[uf.find(i) for i in ix] for _, ix, _, _ in items] + [[uf.find(i) for i in ix] for _, ix, _ in extras]
        else:
            boxes = [list(ix) for _, ix, _ in tn]
        inc = collections.defaultdict(set)
        for b, ix in enumerate(boxes):
            for i in ix:
                if i != 0: inc[i].add(b)   # index 0 = the open output wire
        G = nx.Graph(); G.add_nodes_from(range(len(boxes)))
        for bs in inc.values():
            G.add_edges_from(itertools.combinations(sorted(bs), 2))
        w = treewidth_min_fill_in(G)[0] if G.number_of_edges() else 0
        L = nx.Graph(); L.add_nodes_from(i for ix in boxes for i in ix if i != 0)   # index graph: clique per box (Markov-Shi)
        for ix in boxes:
            L.add_edges_from(itertools.combinations(sorted({i for i in ix if i != 0}), 2))
        _last[0] = treewidth_min_fill_in(L)[0] if L.number_of_edges() else 0; lw[_last[0]] += 1; lw_len[len(tn)].append(lw_last(lw))
        tw[w] += 1; hyp[max((len(b) for b in inc.values()), default=0)] += 1
        # cycles of the wiring itself: hypergraph as bipartite (boxes + wires), cyclomatic = E - V + components
        B = nx.Graph(); B.add_edges_from((('b', b), ('w', i)) for i, bs in inc.items() for b in bs)
        cyc[B.number_of_edges() - B.number_of_nodes() + nx.number_connected_components(B)] += 1; n_box.append(len(boxes))
    res[name] = {'treewidth_minfill': dict(sorted(tw.items())), 'cyclomatic': dict(sorted(cyc.items())),
                 'max_spider_arity': dict(sorted(hyp.items())), 'mean_boxes': sum(n_box) / len(n_box),
                 'index_treewidth_minfill': dict(sorted(lw.items())),
                 'index_treewidth_by_words': {k: [sum(v) / len(v), max(v), len(v)] for k, v in sorted(lw_len.items())}, 'n': len(caps)}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(out_path, 'x'), indent=1)

"""Step 7: the image as a tensor network over its scene graph (vision.method: graph).

Per image (graph row r of data/psg/graphs.pt; tools/tn/s7_psg_graphs.py):
  object j:    leaf o_j = base + P crop_j + E[class_j] in R^D, a copy tensor (index k_j) weighted by o_j
  relation:    (s, o, p) -> R_p = A diag(m_p) B on (k_s, k_o): A = 'a:rel.in', m_p = 'm:<p>', B = 'a:rel.out' (shared A, B;
               identity init, as the step-6 token modifiers), so subject and object legs differ
  apex:        open index z (dim A_out) with leaf g = base_g + P_g CLIP_global ('y:<r>'); every object joins it through the
               shared S = 's:apex' (D x A_out)
  Z(G)[z] = g[z] sum_k prod_j o_j[k_j] S[k_j, z] prod_rel R_p[k_s, k_o]   ->  linear head A_out -> embedding_dim, unit norm
Contracted by the text side's levelwise engine (TNModel): object / global leaves are data leaves ('x:<object row>',
'y:<graph row>') whose values come from data_fn; the other tensors are banks with dense gradients (row_grads False).
vision.relations false: no relation tensors (objects + apex only). vision.global false: g = base_g (no CLIP global vector).
The dataset's image store is data/psg/img_graph.pt ({coco id: [graph row]}), so img_vecs[:, 0] = graph rows."""
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F
from opt_einsum.parser import get_symbol
from modules.models.text.einsum_classical import TNModel, _batched, _level_plan
import os, multiprocessing
from opt_einsum import contract_path


def treewidth(n, edges):
    # min-degree elimination: an upper bound (as tools/tn/s7_psg_graphs.py)
    adj = [set() for _ in range(n)]
    for a, b in edges:
        if a != b: adj[a].add(b); adj[b].add(a)
    alive, tw = set(range(n)), 0
    while alive:
        v = min(alive, key=lambda u: (len(adj[u]), u))
        nb = adj[v]; tw = max(tw, len(nb))
        for a in nb:
            adj[a] |= nb - {a}; adj[a].discard(v)
        alive.discard(v); adj[v] = set()
    return tw


def cap_treewidth(n, rels, max_tw):
    # drop relations until the object graph's treewidth (bound) is <= max_tw: repeatedly remove every relation on the object
    # pair whose endpoints have the largest summed degree. Returns (kept relations, number dropped)
    rels = list(rels)
    while rels and treewidth(n, [(s, o) for s, o, _ in rels]) > max_tw:
        deg = [0] * n
        for a, b in {tuple(sorted((s, o))) for s, o, _ in rels}:
            deg[a] += 1; deg[b] += 1
        pair = max({tuple(sorted((s, o))) for s, o, _ in rels}, key=lambda q: (deg[q[0]] + deg[q[1]], q))
        rels = [r for r in rels if tuple(sorted(r[:2])) != pair]
    return rels


def compile_graph(row, n_obj, obj0, rels, D, A, relations=True):
    # einsum + symbols of one image graph: z = apex index, k_j = object j, r_e = rank index of relation e
    z = get_symbol(0); k = [get_symbol(1 + j) for j in range(n_obj)]
    ins, syms = [z], [{'name': f'y:{row}', 'shape': (A,)}]
    for j in range(n_obj):
        ins += [k[j], k[j] + z]
        syms += [{'name': f'x:{obj0 + j}', 'shape': (D,)}, {'name': 's:apex', 'shape': (D, A)}]
    if relations:
        for e, (s, o, p) in enumerate(rels):
            r = get_symbol(1 + n_obj + e)
            ins += [k[s] + r, r, r + k[o]]
            syms += [{'name': 'a:rel.in', 'shape': (D, D)}, {'name': f'm:{p}', 'shape': (D,)}, {'name': 'a:rel.out', 'shape': (D, D)}]
    return ','.join(ins) + '->' + z, syms


def elim_path(n_obj, rels, relations=True):
    # explicit pairwise contraction path (opt_einsum convention: positions in the current operand list, result appended)
    # for compile_graph's operand order: relation chains first (A m B -> R_e), each object's leaf with S (-> U_j over k_j, z),
    # then objects eliminated in min-degree order of the object graph (z is on every U_j: the apex is kept to the end), so
    # every intermediate stays within the elimination width (+ z). opt_einsum's default search found 1e31-flop paths for
    # some 80-object graphs of treewidth 3.
    ops = [{'z'}]
    for j in range(n_obj):
        ops += [{('k', j)}, {('k', j), 'z'}]
    rels = rels if relations else []
    for e, (s_, o_, _) in enumerate(rels):
        ops += [{('k', s_), ('r', e)}, {('r', e)}, {('r', e), ('k', o_)}]
    path = []
    def merge(i, j):
        i, j = sorted((i, j))
        a, b = ops[i], ops[j]
        rest = set().union(*[ops[t] for t in range(len(ops)) if t not in (i, j)]) | {'z'}
        del ops[j]; del ops[i]
        ops.append((a | b) & rest); path.append((i, j))
    def find(x):
        return [t for t, o in enumerate(ops) if x in o]
    for e in range(len(rels)):   # A_e m_e -> T, T B_e -> R_e
        t = find(('r', e)); merge(t[0], t[1]); t = find(('r', e)); merge(t[0], t[1])
    for j in range(n_obj):   # x_j S_j -> U_j
        t = [u for u in find(('k', j)) if ops[u] in ({('k', j)}, {('k', j), 'z'})][:2]; merge(t[0], t[1])
    adj = [set() for _ in range(n_obj)]
    for s_, o_, _ in rels:
        if s_ != o_: adj[s_].add(o_); adj[o_].add(s_)
    alive = set(range(n_obj))
    while alive:
        v = min(alive, key=lambda u: (len(adj[u]), u))
        while len(find(('k', v))) > 1:   # merge the tensors on k_v, smallest first
            t = sorted(find(('k', v)), key=lambda u: len(ops[u]))
            merge(t[0], t[1])
        t = find(('k', v))
        if t:   # one tensor left holding k_v: merge it with a neighbour's tensor, else with the apex accumulator
            nb = [u for u in range(len(ops)) if u != t[0] and (ops[u] & ops[t[0]]) - {'z'}]
            acc = [u for u in range(len(ops)) if u != t[0] and ops[u] == {'z'}]
            merge(t[0], (nb or acc)[0])
        for a in adj[v]:
            adj[a] |= adj[v] - {a}; adj[a].discard(v)
        alive.discard(v); adj[v] = set()
    while len(ops) > 1:
        merge(0, 1)
    return path


def _graph_plan(args):
    e, shapes, path = args
    b = _batched(e)
    _, info = contract_path(b, *[(64, *sh) for sh in shapes], optimize=path, shapes=True)
    return (b, shapes), path, _level_plan(b, shapes, info.contraction_list), float(info.opt_cost)


class ImageGraphTN(nn.Module):
    def __init__(self, embedding_dim, graph_path='data/psg/graphs.pt', crops_path='data/psg/crops_b32.pt', graph_dim=32,
                 apex_dim=128, relations=True, use_global=True, scale_free=True, rows=None, max_tw=None):
        # rows: compile only these graph rows (checks); None = all
        super().__init__()
        G = torch.load(graph_path)
        self.D, self.A, self.relations, self.use_global = graph_dim, apex_dim, relations, use_global
        n_cls, clip_dim = len(G['class_names']), G['glob'].shape[1]
        # frozen inputs: buffers, not saved in checkpoints (persistent=False; rebuilt from the data files)
        self.register_buffer('crops', F.normalize(torch.load(crops_path).float(), dim=-1).half(), persistent=False)
        self.register_buffer('cls', G['obj_cls'].long(), persistent=False)
        self.register_buffer('glob', F.normalize(G['glob'].float(), dim=-1), persistent=False)
        self.P = nn.Linear(clip_dim, graph_dim, bias=False)
        self.E = nn.Embedding(n_cls, graph_dim)
        self.base = nn.Parameter(torch.randn(graph_dim) / graph_dim ** 0.5)
        self.P_g = nn.Linear(clip_dim, apex_dim, bias=False)
        self.base_g = nn.Parameter(torch.randn(apex_dim) / apex_dim ** 0.5)
        nn.init.normal_(self.E.weight, std=graph_dim ** -0.5)
        start = G['obj_start'].tolist()
        keep = set(range(len(start) - 1)) if rows is None else set(rows)
        # max_tw (vision.max_tw): relations dropped until each object graph has treewidth <= max_tw (memory: intermediates
        # grow as D^(tw+1) x A); counts kept in self.capped
        rel = G['rel']
        if max_tw is not None and relations:
            rel = [cap_treewidth(start[r + 1] - start[r], G['rel'][r], max_tw) if int(G['tw'][r]) > max_tw else G['rel'][r] for r in range(len(start) - 1)]
        self.capped = {'graphs': sum(len(a) != len(b) for a, b in zip(rel, G['rel'])), 'relations_dropped': sum(len(b) - len(a) for a, b in zip(rel, G['rel'])),
                       'relations': sum(map(len, G['rel']))}
        if max_tw is not None: print(f'ImageGraphTN: max_tw {max_tw}: {self.capped}')
        self.rel = rel
        self.graphs = [compile_graph(r, start[r + 1] - start[r], start[r], rel[r], graph_dim, apex_dim, relations)
                       if r in keep else None for r in range(len(start) - 1)]
        self.tw, self.tw_apex = G['tw'], G['tw_apex']
        eng = TNModel(out_dim=apex_dim, engine='levelwise', init='glorot', scale_free=scale_free, head='plain')
        eng.row_grads, eng.data_fn = False, self._data
        # data leaves first, in data-row order, so that bank row = object row / graph row
        order = [[{'name': f'x:{j}', 'shape': (graph_dim,)} for j in range(start[-1])],
                 [{'name': f'y:{r}', 'shape': (apex_dim,)} for r in range(len(self.graphs))]]
        eng.from_symbols(order + [g[1] for g in self.graphs if g is not None])
        with torch.no_grad():
            for name in eng.symbols:
                b, r = eng.sym_loc[name]
                if name in ('a:rel.in', 'a:rel.out'): eng._bank(b)[r].copy_(torch.eye(graph_dim))
                elif name.startswith('m:'): eng._bank(b)[r].fill_(1.0)
                elif name == 's:apex': nn.init.normal_(eng._bank(b)[r], std=graph_dim ** -0.5)
        # explicit elimination-order paths (elim_path), searched by a fork pool, then the engine's caches
        todo = [(e, tuple(tuple(d['shape']) for d in sy), elim_path(start[r + 1] - start[r], rel[r], relations))
                for r, (e, sy) in ((r, g) for r, g in enumerate(self.graphs) if g is not None)]
        if len(todo) > 2000:
            with multiprocessing.get_context('fork').Pool(int(os.environ.get('TN_PATH_WORKERS', 0)) or len(os.sched_getaffinity(0))) as pool:
                found = pool.map(_graph_plan, todo, chunksize=256)
        else:
            found = [_graph_plan(t) for t in todo]
        self.flops = {}
        for key, path, plan, cost in found:
            eng.path_cache[key] = path; eng.level_cache[key] = plan; self.flops[key[0]] = cost
        self.engine = eng
        self.head = nn.Linear(apex_dim, embedding_dim, bias=False)

    def _data(self, kind, rows):
        if kind == 'x':
            return self.base + self.P(self.crops.index_select(0, rows).float()) + self.E(self.cls.index_select(0, rows))
        if self.use_global:
            return self.base_g + self.P_g(self.glob.index_select(0, rows))
        return self.base_g.expand(len(rows), -1)

    def contract(self, batch):
        # batch: list of (einsum, symbols); returns the apex vectors (pre-head)
        eng = self.engine
        if not hasattr(eng, '_plans'):
            eng._plans, eng._struct_ids, eng._structs = {}, {}, []
        groups = defaultdict(list)
        for i, (e, syms) in enumerate(batch):
            sid, rows = eng._plan(e, syms)
            groups[sid].append((i, rows))
        return eng._levelwise(groups)

    def forward(self, img_vecs):
        rows = img_vecs[:, 0].round().long().tolist()
        out = self.contract([self.graphs[r] for r in rows])
        return F.normalize(self.head(out), dim=-1, eps=1e-12)

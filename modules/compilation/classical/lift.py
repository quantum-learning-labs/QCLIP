"""E2: Frobenius lifting of the decoupled compile (design ~/.claude/plans/tn-e2-frobenius.md, approved 2026-11-11).

A word keeps ONE core meaning; the higher-order grammatical roles of that word are built from it with parameter-free
Frobenius copy / merge maps (spiders). In the einsum a spider is just a shared index (a hyperedge): every leg in one spider
gets the same index. Rules (each checked on the training wiring, results/tn/step5/audit/README.md), by the word's leg
signature after R1 and its coarse wiring pattern (which legs share a neighbour word, '*' = the caption output):
  R1 root independence: no type contains 'out'; the root's free wire gets its atomic type X back (per-word table from the
     training parses, tools/tn/s5_e2_rootmap.py) and is closed by one shared codomain map 'm:X->out' per atomic type.
  R2 VP-modifying preposition  s@n@n@s@n AABBC / AAB*C (also object p: s@n@n@s@p):
     legs (s_vp, n_vp, n_res, s_res, obj) -> subject spider {n_res, n_vp, core.in}; s spider {s_vp, s_res, m:n->s(core.out)};
     the word's core is its noun-modifier type 't:n@n@<obj>' (the SAME tensor as "on" in "a cat on a mat").
  R2b VP turned noun modifier  s@n@n@n@n AABCD: legs (s_vp, n_vp, n_in, n_out, obj) -> n spider {n_in, n_vp, core.in, n_out,
     m:s->n(.)}; s spider {s_vp, m:n->s(core.out), m:s->n.in}  (R2 then R4).
  R3 auxiliary / copula        n@s@s@n AABB / A*BB: legs (n_subj, s_res, s_arg, n_arg) -> {n_subj, n_arg} wired through;
     the word keeps an order-2 core 't.aux:s@s' on (s_arg -> s_res).
  R4 relative pronoun          n@n@s@n ABCC: legs (n_in, n_out, s_vp, n_vp) -> one n spider {n_in, n_out, n_vp,
     m:s->n(s_vp)}; the word itself carries no tensor (Sadrzadeh, Clark & Coecke 2013).
  R5 coordination of modifiers n@n@n@n@n@n AABCDD: legs (l_arg, l_res, n_out, n_in, r_arg, r_res) -> copy {n_in, l_arg, r_arg},
     merge {l_res, r_res} -> word core 't.conj:n@n' -> n_out.
     of verb phrases           s@n@n@s@s@n AABBCC / AAB*CC: legs (l_s, l_n, n_res, s_res, r_s, r_n) -> copy {n_res, l_n, r_n},
     merge {l_s, r_s} -> word core 't.conj:s@s' -> s_res.
  R6 adverb on a VP            s@n@n@s AABB / s@n@n@out AAB*: legs (s_arg, n_arg, n_res, s_res) -> {n_arg, n_res} wired
     through; word core 't.adv:s@s' on (s_arg -> s_res).
     modifier of an adjective  n@n@n@n ABCC: legs (n_out, n_in, adj_arg, adj_res) -> {n_in, adj_arg} wired through; word core
     't.adv:n@n' on (adj_res -> n_out).
An occurrence whose wiring does not match its rule's pattern, or a root word with no R1 entry, is left as it is.
"""
from __future__ import annotations
import json
from collections import defaultdict, Counter

RULES = ('R1', 'R2', 'R3', 'R4', 'R5', 'R6')


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        while p.get(x, x) != x:
            p[x] = p.get(p[x], p[x]); x = p[x]
        return x

    def union(self, *xs):
        r = self.find(xs[0])
        for x in xs[1:]:
            q = self.find(x)
            if q == r: continue
            if q == 0: self.p[r] = 0; r = 0   # the caption output index stays 0
            else: self.p[q] = r


def coarse_patterns(tn):
    # per word: one letter per leg naming the neighbour word that leg is contracted with (A, B, ... in leg order), '*' = free
    ends = defaultdict(list)
    for wi, (_, idx, _) in enumerate(tn):
        for li, i in enumerate(idx): ends[i].append((wi, li))
    pats = []
    for wi, (_, idx, _) in enumerate(tn):
        letters, s = {}, []
        for li, i in enumerate(idx):
            other = [w for w, l in ends[i] if (w, l) != (wi, li)]
            s.append('*' if not other else letters.setdefault(other[0], 'ABCDEFGHIJKL'[len(letters)]))
        pats.append(''.join(s))
    return pats


def load_rootmap(path):
    with open(path) as f:
        rm = json.load(f)
    # fallback free-wire type: the most common X over the per-word entries
    rm['fallback'] = Counter(rm['by_word'].values()).most_common(1)[0][0]
    return rm


def lift(tn, rules, rootmap, counter):
    """tn: [(word, idx, types)] as stored (after unify_codomain). Returns (items, extras, uf, stats):
    items  [(word, idx, types, type_symbol)]: words that still carry a word vector + type tensor (type_symbol None = default
           't:'+sig), extras [(symbol, idx, types)]: parameter tensors without a word (codomain and s<->n maps),
    uf the spider union-find over indices (apply uf.find to every index of every emitted tensor)."""
    rules = set(rules)
    pats = coarse_patterns(tn)
    uf, items, extras, stats = _UF(), [], [], defaultdict(int)
    for (word, idx, types), pat in zip(tn, pats):
        idx, types = list(idx), list(types)
        root = 'out' in types
        if root:
            k = types.index('out')
            key = f"{word.lower()}|{'@'.join(types)}"
            X = (rootmap['by_word'].get(key) or rootmap['global'].get('@'.join(types))) if 'R1' in rules else None
            if X is None and 'R1' in rules:
                # no evidence for this root signature (0.01% of training roots): the most common free-wire type, so that EVERY
                # caption leaves through one m:X->out map (no second output path through a root type's own 'out' leg)
                X = rootmap['fallback']; stats['R1_fallback'] += 1
            if X is None:
                items.append((word, idx, types, None)); stats['root_kept'] += 1; continue
            r = next(counter)
            extras.append((f'm:{X}->out', [r, idx[k]], [X, 'out']))
            idx[k], types[k] = r, X
            stats['R1'] += 1
        sig = '@'.join(types)
        P = pat   # computed on the stored wiring: the root's output leg is free ('*'), as the rules' root forms are written
        if 'R2' in rules and sig in ('s@n@n@s@n', 's@n@n@s@p') and P[0] == P[1] and len({P[0], P[2], P[4]}) == 3 \
                and (P[2] == P[3] or P[3] == '*') and P[2] != '*':
            a, b, c, d, e = idx; y = next(counter)
            uf.union(c, b); uf.union(a, d)
            items.append((word, [c, y, e], ['n', 'n', types[4]], 't:n@n@' + types[4]))
            extras.append(('m:n->s', [y, d], ['n', 's'])); stats['R2'] += 1; continue
        if 'R2' in rules and 'R4' in rules and sig == 's@n@n@n@n' and P[0] == P[1] and len(set(P[0] + P[2:])) == 4 and '*' not in P:
            a, b, c, d, e = idx; y, z = next(counter), next(counter)
            uf.union(c, b); uf.union(c, d); uf.union(c, z)
            items.append((word, [c, y, e], ['n', 'n', 'n'], 't:n@n@n'))
            extras += [('m:n->s', [y, a], ['n', 's']), ('m:s->n', [a, z], ['s', 'n'])]; stats['R2b'] += 1; continue
        if 'R3' in rules and sig == 'n@s@s@n' and P[2] == P[3] and P[2] not in (P[0], P[1], '*') \
                and (P[0] == P[1] or P[1] == '*') and P[0] != '*':
            x, r_, a, b = idx
            uf.union(x, b)
            items.append((word, [a, r_], ['s', 's'], 't.aux:s@s')); stats['R3'] += 1; continue
        if 'R4' in rules and sig == 'n@n@s@n' and P[2] == P[3] and len({P[0], P[1], P[2]}) == 3 and '*' not in P:
            i, o, s_, v = idx; z = next(counter)
            uf.union(i, o); uf.union(i, v); uf.union(i, z)
            extras.append(('m:s->n', [s_, z], ['s', 'n'])); stats['R4'] += 1; continue
        if 'R5' in rules and sig == 'n@n@n@n@n@n' and P[0] == P[1] and P[4] == P[5] and len({P[0], P[2], P[3], P[4]}) == 4 and '*' not in P:
            la, lr, o, i, ra, rr = idx
            uf.union(i, la); uf.union(i, ra); uf.union(lr, rr)
            items.append((word, [lr, o], ['n', 'n'], 't.conj:n@n')); stats['R5n'] += 1; continue
        if 'R5' in rules and sig == 's@n@n@s@s@n' and P[0] == P[1] and P[4] == P[5] and (P[2] == P[3] or P[3] == '*') \
                and len({P[0], P[2], P[4]}) == 3 and P[2] != '*':
            ls, ln, nr, sr, rs, rn = idx
            uf.union(nr, ln); uf.union(nr, rn); uf.union(ls, rs)
            items.append((word, [ls, sr], ['s', 's'], 't.conj:s@s')); stats['R5s'] += 1; continue
        if 'R6' in rules and sig == 's@n@n@s' and P[0] == P[1] and (P[2] == P[3] or P[3] == '*') and P[2] not in (P[0], '*'):
            a, b, c, d = idx
            uf.union(b, c)
            items.append((word, [a, d], ['s', 's'], 't.adv:s@s')); stats['R6s'] += 1; continue
        if 'R6' in rules and sig == 'n@n@n@n' and P[2] == P[3] and len({P[0], P[1], P[2]}) == 3 and '*' not in P:
            o, i, aa, ar = idx
            uf.union(i, aa)
            items.append((word, [ar, o], ['n', 'n'], 't.adv:n@n')); stats['R6n'] += 1; continue
        items.append((word, idx, types, None))
    return items, extras, uf, stats

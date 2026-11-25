"""Step 6 (text.tokens: bpe): words as token sub-networks. Segmentation = CLIP's byte-pair vocabulary
(clip.simple_tokenizer; only the segmentation is used, no CLIP vectors). A word with pieces t1..tn is the chain

    w = M(tn) ... M(t2) v(t1),   M(t) = A diag(m_t) B

on its word wire: the root piece's vector v(t1) ('w:<piece>-', a word-bank vector: word_base, dropout, centre apply), one
modifier vector m_t per piece ('k:<piece>', bank of its own) and two shared d x d factors A = 'a:tok.in', B = 'a:tok.out'
(the rank leg is a hyperedge of A, m_t and B, as in the CP type tensors). One-piece words keep their symbol 'w:<word>', so
their compile is unchanged. Init (TNModel.tokens_reset): A = B = I and m = 1, every modifier is the identity (an isometry):
a word starts as its root vector and the chain has gain 1 at any length. Unseen pieces at test: root random (as any OOV
word), modifier 1 (neutral)."""


class BPEPieces:
    def __init__(self):
        from clip.simple_tokenizer import SimpleTokenizer
        self.tok, self.cache = SimpleTokenizer(), {}

    def __call__(self, word):
        ps = self.cache.get(word)
        if ps is None:
            ps = self.cache[word] = tuple(self.tok.decoder[i] for i in self.tok.encode(word))
        return ps


def word_chain(word, wire, dim, pieces, index_counter):
    # (symbol, indices, shape) of the tensors of one word whose output wire is `wire`
    ps = pieces(word)
    if len(ps) <= 1:
        return [('w:' + word, [wire], (dim,))]
    x = next(index_counter)
    out = [('w:' + ps[0] + '-', [x], (dim,))]
    for j, p in enumerate(ps[1:]):
        r = next(index_counter)
        y = wire if j == len(ps) - 2 else next(index_counter)
        out += [('a:tok.in', [x, r], (dim, dim)), ('k:' + p, [r], (dim,)), ('a:tok.out', [r, y], (dim, dim))]
        x = y
    return out

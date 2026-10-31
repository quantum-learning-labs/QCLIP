"""Step-4b collapse check: the text head (LayerNorm gamma, beta) of checkpoints. ||beta||, ||gamma||, the cosine between unrelated
captions predicted by a shared offset, ||beta||^2/(||beta||^2+||gamma||^2) (z has unit variance per coordinate, ||gamma*z||~||gamma||),
cos(beta, mean image embedding), and the projection of each image onto beta vs its hubness rank (from s4_embed json if given).
  s4_head.py <ckpt.pt>..."""
import sys, torch
emb = torch.load('data/mscoco/processed/image_embeddings.pt', weights_only=False)
X = torch.nn.functional.normalize(torch.stack(list(emb.values())).float(), dim=1)
mu = X.mean(0)
print(f"mean image embedding norm (of unit vectors): {mu.norm():.3f}")
for p in sys.argv[1:]:
    sd = torch.load(p, map_location='cpu', weights_only=False)['text']
    g, b = sd['norm_head.weight'].float(), sd['norm_head.bias'].float()
    pred = float(b.norm() ** 2 / (b.norm() ** 2 + (g ** 2).sum()))
    proj = X @ torch.nn.functional.normalize(b, dim=0)
    print(f"{p.split('/')[-2]:22s} |gamma| {g.norm():6.2f} |beta| {b.norm():6.2f} predicted cos(unrelated) {pred:.3f} "
          f"cos(beta, mean img) {torch.nn.functional.cosine_similarity(b, mu, dim=0):.3f} img.proj on beta mean {proj.mean():.3f} max {proj.max():.3f}")

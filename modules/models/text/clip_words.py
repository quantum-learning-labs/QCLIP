"""text.word_init: clip (step 5, a temporary stand-in: the goal is no reliance on pretrained models). Each vocabulary word is
encoded alone by CLIP ViT-B/32's text tower; the unit vectors are centred and projected to `dim` by PCA (eigendecomposition of the
covariance on the CPU: CUDA eigh is broken in our envs). Returns {word: vector}."""
import torch, clip

@torch.no_grad()
def clip_word_vectors(words, dim, device='cuda'):
    model, _ = clip.load('ViT-B/32', device=device); model.eval()
    out = []
    for j in range(0, len(words), 1024):
        tok = clip.tokenize(words[j:j + 1024], truncate=True).to(device)
        out.append(torch.nn.functional.normalize(model.encode_text(tok).float(), dim=1).cpu())
    X = torch.cat(out).double(); X = X - X.mean(0)
    evals, evecs = torch.linalg.eigh(X.T @ X / len(X))
    P = evecs[:, evals.argsort(descending=True)[:dim]]
    Y = (X @ P).float()
    del model; torch.cuda.empty_cache()
    return dict(zip(words, Y))

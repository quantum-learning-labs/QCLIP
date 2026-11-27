"""encode / per_caption from s5e_gap_breakdown.py (importable: that tool runs at import)."""
import torch
import numpy as np

def encode(ev, loader, mapper):
    imgs, txts, img_of, caps, gi = [], [], [], [], 0
    for batch in loader:
        img_batch, txt_batch = mapper(batch)
        imgs.append(ev._encode_img(img_batch)); pay = []
        for cs in txt_batch:
            if isinstance(cs[0], str) or isinstance(cs[0][0], dict): cs = [cs]
            for c in cs: img_of.append(gi); pay.append((c[0], c[1])); caps.append((c[0], c[1]))
            gi += 1
        txts.append(ev._encode_txt(pay))
    return torch.cat(imgs), torch.cat(txts), np.array(img_of), caps

def per_caption(ev, I, C, img_of):
    S = ev._readout(C @ I.conj().T)   # captions x images
    S = S.nan_to_num(nan=float('-inf'))
    io = torch.from_numpy(img_of).to(S.device)
    pos = S.gather(1, io.view(-1, 1)); neg = torch.ones_like(S, dtype=torch.bool); neg[torch.arange(len(S)), io] = False
    g = ((S > pos) & neg).sum(1).double(); t = ((S == pos) & neg).sum(1).double()
    return {f'R{k}': ((k - g) / (t + 1)).clamp(0, 1).cpu().numpy() for k in (1, 5, 10)}


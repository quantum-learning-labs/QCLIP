# L4: on the swap task (one image, two captions), does the Pauli-cosine ranking (what Pauli_InfoNCE trains) agree
# with Tr(rho sigma) (what the swap test / benchmark measures)? usage: swap_rescore.py <out.json> <ckpt_dir>...
import sys, json, yaml, itertools, torch, pandas as pd
from pathlib import Path
from modules.utils.factory import build_experiment
from modules.utils.general import set_seed, CheckpointManager
from modules.data_pipeline.engine import DataEngine
out, ckpts = sys.argv[1], sys.argv[2:]
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
P1 = [torch.eye(2), torch.tensor([[0, 1], [1, 0]]), torch.tensor([[0, -1j], [1j, 0]]), torch.tensor([[1, 0], [0, -1]])]
def pauli_basis(n):
    B = []
    for t in itertools.product(range(4), repeat=n):
        if all(i == 0 for i in t): continue
        m = P1[t[0]].to(torch.complex64)
        for i in t[1:]: m = torch.kron(m, P1[i].to(torch.complex64))
        B.append(m)
    return torch.stack(B).to(DEV)
res = {}
for ck in map(Path, ckpts):
    cfg = yaml.safe_load(open(ck / 'config.yaml')); set_seed(cfg['seed']); assert cfg['text'].get('discard')
    ans, img, txt, _ = build_experiment(cfg, DEV); de = DataEngine(cfg, ans, DEV)
    de.text_init(txt, pd.concat([de.compile_text('train'), de.compile_text('val')], ignore_index=True)); de.image_init(img)
    img.to(DEV); txt.to(DEV); CheckpointManager.load_model(ck / 'best.pt', img, txt, DEV); img.to(DEV); txt.to(DEV); img.eval(); txt.eval()
    PB = pauli_basis(cfg['embedding_qubits']); d = 2 ** cfg['embedding_qubits']
    comp = de.compile_text('swap'); loader = de.get_loader(comp, split='swap')
    tr_c, cos_c, agree, n = 0, 0, 0, 0
    with torch.no_grad():
        for b in loader:
            sig = img(b['image'].to(DEV)).reshape(-1, d, d); rp = txt(b['pos_caption']).reshape(-1, d, d); rn = txt(b['neg_caption']).reshape(-1, d, d)
            tr = lambda r: torch.einsum('bij,bji->b', r, sig).real
            vec = lambda r: torch.einsum('bmn,knm->bk', r, PB).real
            cos = lambda r: torch.nn.functional.cosine_similarity(vec(r), vec(sig), dim=1)
            a, c = tr(rp) > tr(rn), cos(rp) > cos(rn)
            tr_c += a.sum().item(); cos_c += c.sum().item(); agree += (a == c).sum().item(); n += len(a)
    res[ck.name] = dict(pairs=n, acc_tr=tr_c / n, acc_pauli_cos=cos_c / n, ranking_agreement=agree / n)
    print(ck.name, res[ck.name], flush=True)
json.dump(res, open(out, 'x'), indent=1)

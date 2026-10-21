import math
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from modules.utils.factory import build_experiment
from modules.utils.general import CheckpointManager
from modules.data_pipeline.engine import DataEngine


def _as_dm(emb: torch.Tensor) -> torch.Tensor:
    # [B, D, D] density matrix from a density matrix ([B, D, D] or flattened [B, D*D]) or a state vector ([B, 2, 2, ...] / [B, D])
    B = emb.size(0)
    if emb.dim() == 3 and emb.size(1) == emb.size(2):
        return emb.to(torch.complex64)
    v = emb.reshape(B, -1).to(torch.complex64)
    D = int(round(math.sqrt(v.size(1))))
    if emb.dim() == 2 and D * D == v.size(1) and D > 1 and (D & (D - 1)) == 0 and v.size(1) > 8:   # flattened density matrix
        return v.view(B, D, D)
    v = F.normalize(v, dim=1)
    return v[:, :, None] * v.conj()[:, None, :]


def _as_state(emb: torch.Tensor) -> torch.Tensor:
    # normalised [B, D] state vector of a pure teacher embedding
    return F.normalize(emb.reshape(emb.size(0), -1).to(torch.complex64), dim=1)


def state_fidelity(teacher_vec: torch.Tensor, student_emb: torch.Tensor) -> torch.Tensor:
    # <psi_T| rho_S |psi_T> / Tr rho_S per row: 1 iff the student state equals the teacher's (pure) post-selected state
    rho = _as_dm(student_emb)
    tr = rho.diagonal(dim1=1, dim2=2).sum(-1).real.clamp(min=1e-12)
    return torch.einsum('bi,bij,bj->b', teacher_vec.conj(), rho, teacher_vec).real / tr


class Teacher:
    """Frozen post-selected (non-discard) teacher. Its circuits compile differently from the student's, so its
    text and image embeddings are computed once for every training row and looked up by row index."""

    def __init__(self, ckpt_path, student_config, student_train_df, device, kd_weight=1.0, kd_tau=1.0, state_weight=0.0):
        self.kd_weight, self.kd_tau, self.state_weight = kd_weight, kd_tau, state_weight
        cfg = yaml.safe_load(open(Path(ckpt_path).parent / 'config.yaml'))
        cfg['num_workers'] = 0
        ansatz, image_model, text_model, loss_fn = build_experiment(cfg, device)
        engine = DataEngine(cfg, ansatz, device)
        train_df, val_df = engine.compile_text('train'), engine.compile_text('val')
        if len(train_df) != len(student_train_df) or not (train_df['pos_image_id'].values == student_train_df['pos_image_id'].values).all():
            raise RuntimeError('teacher and student training rows are not aligned (row count or image ids differ)')
        import pandas as pd
        engine.text_init(text_model, pd.concat([train_df, val_df], ignore_index=True))
        engine.image_init(image_model)
        CheckpointManager.load_model(ckpt_path, image_model, text_model, device)
        image_model.to(device).eval(); text_model.to(device).eval()
        self.loss_fn = loss_fn.to(device)
        loader = DataLoader(engine.get_dataset(train_df, split='train'), batch_size=cfg['batch_size'], shuffle=False,
                            collate_fn=engine.collate_fn, num_workers=0)
        txt, img = [], []
        with torch.no_grad():
            for batch in loader:
                img.append(image_model(batch['pos_image'].to(device)))
                txt.append(text_model(batch['caption']))
        self.text = torch.cat(txt); self.image = torch.cat(img)
        self.text_vec, self.image_vec = _as_state(self.text), _as_state(self.image)
        print(f"Teacher {ckpt_path}: {len(self.text)} rows | KD weight {kd_weight} tau {kd_tau} | state weight {state_weight}")

    def loss(self, idx, student_loss_fn, text_emb, image_emb):
        out = torch.zeros((), device=text_emb.device)
        if self.kd_weight:
            with torch.no_grad():
                t = self.loss_fn.compute_similarity(self.text[idx], self.image[idx]) * self.loss_fn.scale().to(text_emb.device)
            s = student_loss_fn.compute_similarity(text_emb, image_emb) * student_loss_fn.scale().to(text_emb.device)
            tau = self.kd_tau
            kd = lambda S, T: F.kl_div(F.log_softmax(S / tau, dim=1), F.log_softmax(T / tau, dim=1), log_target=True, reduction='batchmean')
            out = out + self.kd_weight * tau ** 2 * 0.5 * (kd(s, t) + kd(s.T, t.T))
        if self.state_weight:
            fid = 0.5 * (state_fidelity(self.text_vec[idx], text_emb) + state_fidelity(self.image_vec[idx], image_emb))
            out = out + self.state_weight * (1 - fid).mean()
        return out

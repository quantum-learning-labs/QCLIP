# TN retrieval, step 1: baseline and pipeline check (2026-10-23)

Classical tensor-network text encoder (model_type `tn`) trained on MSCOCO, then evaluated zero-shot on the
compositional benchmarks. Tables (mean ± std over seeds 0-2, chance, z vs chance, OOV): `step1_tables.md`.

## Frozen reference (every later step compares against exactly this)
- Config `configs/tn/ref_clip.yaml`, seeds 0, 1, 2, training code `ccd1411`.
- Text: TNCompiler + MPS (n=s=p=128, out=512, bond_dim 8, max_order 1), TNModel (LayerNorm + L2 head). 537.9M params.
- Image: frozen CLIP ViT-B/32 vectors (`data/mscoco/processed/image_embeddings.pt`), L2-normalised (`FrozenCLIP(normalize=True)`).
- InfoNCE T=0.07 without abs (real dot products; `TNModel.real_scores`), Adam lr 1e-3, wd 1e-4, batch 512,
  3 epochs (1 warmup + cosine), `last.pt` reported.
- Data: train = MSCOCO train2017 minus a 5k-image holdout (seed 0, `data/mscoco/processed/tn/`), val = holdout
  (i2tR1 monitored), test coco5k = val2017.
- Runs `tn_s1_clip_h_s{0,1,2}` (HPC: L40S, L40S, RTX 6000 Ada). Their per-step losses: `checkpoints/mscoco/tn/<run>/steps.csv`.

## Headline numbers (reference)
coco5k i2t R@1/5/10 0.0065/0.0251/0.0414, t2i 0.0011/0.0047/0.0088 (above chance); ARO rel/att, SVO std/swap,
SugarCREPE++ (5 subsets) at chance (|z| <= 3.5); Winoground txt 0.145, img 0.151, grp 0.064 (below chance).
Train loss ends at ~ln(512) = 6.24: 3 epochs barely trains the model.
The dropped TTN image tower (`ref_ttn.yaml`, runs `tn_s1_ttn_p_s*`) stayed at chance on MSCOCO.

## Faults found / fixed
- tn.yaml dims with bond 32 / max_order 2: 7.72B params on full MSCOCO (113k word@type tensors, 48% used once).
- Pipeline fixes: config schema, TTN optimizer + transform, grad_norm_by_type and describe_einsum for TN, scheduler
  T_max = 0 crash, benchmark mapper for global_retrieval, OOV crash (now seeded by (seed, name), rates vs trained vocab),
  .pt keys from COCO file names, retriever subsets, md5 ids in embed_images.
- Text LayerNorm head gradient 40-100x the TN tensors' (step 4).
- Winoground: 125-142/390 caption pairs embed with cos > 0.99, 71-76% of items caption-dominated (`tools/tn/wino_diag.py`).
- Determinism: same seed and GPU = bit-identical losses; legacy path-cache key identical too. Across GPU types step
  losses drift up to 2e-3, so regression checks must use one GPU type.

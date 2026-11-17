# Step 5c: changing what the contrastive loss sees (report, 17 Nov 2026)

## 1. Answer
Two changes inside the contrastive regime move the TN past the bar of the MLP tree model (~320, same tree + frozen CLIP
B/32 image vector):
- **A learned logit scale.** Every TN run since step 4 used InfoNCE with a fixed temperature 0.07. Letting the scale learn
  (InfoNCE + learnable temperature) gives rsum 329.9; SigLIP (pairwise sigmoid, learned scale and bias) gives **335.2 raw**
  (333.9 / 336.6), 346-349 with CSLS. SVO reaches its target (0.849), Winoground text rises to 0.206.
- **Hard negatives built from the diagrams** (same wiring and types, different word assignment; no re-parse). ARO-Att
  0.803 (target 0.70 met), ARO-Rel 0.585, swap_full 0.995, SC++ swap-attribute 0.470, at a recall cost.
- **SigLIP + hard negatives: rsum 318.9** (317.4 / 320.5), ARO-Att 0.755, ARO-Rel 0.575, SVO 0.841, swap_full 0.995,
  SC++ replace-relation 0.443 (best of any TN). One configuration now meets the ARO-Att and (nearly) the SVO targets at
  the ~320 recall level; SigLIP alone is the recall model.
- The fixed temperature also caused most of the hubness: CSLS adds +30-37 rsum to fixed-temperature models but only
  +12-15 to SigLIP / learnable-temperature models.

## 2. Setup
Base: the step-5b recipe (CP128 + R1 codomain maps, EMA 0.999, constant LR 1e-3, batch 512, 150 epochs, best.pt by val
MRR). 2 seeds each. Code tn-step5 a3787b0 (SigLIP, negatives) and f53eb12 (learnable temperature).
- SigLIP: L = -(1/B) sum_ij log sigmoid(z_ij (t <x_i, y_j> + b)), z = +1 on the diagonal, -1 off; t, b learned, init 10, -10.
- Hard negatives (tools/tn/s5_make_negs.py -> data/mscoco/processed/tn/train_negs.pkl, a new file): one per training
  caption, kind drawn uniformly among those available: noun swap (57%), relation replacement on <-> under, with <->
  without, ... (34%), modifier swap between different nouns (9%); 99.7% of 564,450 captions have one. Each image also
  scores the batch's negative captions (extra softmax columns, or extra -1 pairs under SigLIP).
- Hard negatives seed 1 (InfoNCE) hit the cluster's 14 h limit at epoch 131 on a slow node; EMA runs cannot be resumed
  (the EMA shadow is not in the saved state), so its row uses best.pt by val MRR up to epoch 131.

## 3. Results (test, mean of 2 seeds)
| model | seeds | rsum | ARO-Rel | ARO-Att | SVO | swap_full | SC++ repl-obj | SC++ repl-att | SC++ repl-rel | SC++ swap-obj | SC++ swap-att | Wino text | Wino image |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EMA recipe (InfoNCE, fixed T 0.07) | 2 | 288.1 | 0.520 | 0.570 | 0.799 | 0.950 | 0.772 | 0.466 | 0.381 | 0.357 | 0.315 | 0.144 | 0.067 |
| InfoNCE, learnable temperature | 2 | 329.9 | 0.517 | 0.568 | 0.844 | 0.933 | 0.759 | 0.516 | 0.425 | 0.357 | 0.375 | 0.192 | 0.068 |
| SigLIP | 2 | 335.2 | 0.522 | 0.569 | 0.849 | 0.924 | 0.747 | 0.519 | 0.413 | 0.371 | 0.381 | 0.206 | 0.092 |
| hard negatives (InfoNCE, fixed T) | 2 | 272.2 | 0.585 | 0.803 | 0.805 | 0.995 | 0.714 | 0.449 | 0.420 | 0.416 | 0.470 | 0.106 | 0.082 |
| SigLIP + hard negatives | 2 | 318.9 | 0.575 | 0.755 | 0.841 | 0.995 | 0.705 | 0.469 | 0.443 | 0.410 | 0.447 | 0.153 | 0.094 |

Targets: rsum >= 370, ARO-Att >= 0.70, ARO-Rel >= 0.60, SVO >= 0.85, SC++ >= 0.75 per subset, Winoground >= 0.287.

## 4. CSLS readout (k = 10, alpha = 0.5, COCO-val bank; results/tn/step5/audit/wino_hub_5c.json)
| model | raw rsum | CSLS | gain |
|---|---|---|---|
| EMA recipe (fixed T) | 288.1 | 318.0 | +29.9 |
| hard negatives (fixed T) | 272.2 | 309.6 | +37.3 |
| InfoNCE, learnable temperature | 329.9 | 345.3 | +15.3 |
| SigLIP | 335.2 | **347.4** | +12.1 |
| SigLIP + hard negatives | 318.9 | 332.3 | +13.3 |
Winoground stays below chance under CSLS for every model (best: SigLIP text 0.215, image 0.113).

## 5. What it says
- The temperature, not the architecture, held recall back since step 4: one scalar is worth +42 rsum. The sigmoid adds about +5 and helps Winoground
  (text 0.206 vs 0.192, image 0.092 vs 0.068).
- Learning the scale also removes most of the hubness that CSLS was correcting: hubness here was a symptom of a badly
  scaled softmax, not of the representation.
- Hard negatives fix exactly what the step-5 diagnosis predicted: the objective, not the capacity, drained relation and
  attribute words (compare E1: own tensors gained ARO-Rel +0.024 at -53 rsum; negatives gain ARO-Rel +0.053 and ARO-Att
  +0.19 at -16 rsum under SigLIP). Modifier swaps are close to the ARO-Att test format, so part of that gain is format match.
- Not moved: Winoground (below chance everywhere), SC++ replace-object (drops with negatives), ARO-Rel stays under 0.60
  (63% of ARO-Rel is left/right, which the pooled image vector likely does not encode; step 5 section 6).

## 6. Next
1. Rebalance the negatives toward relations (noun swaps are 57% now) and try a negative weight < 1 under SigLIP, to keep
   SigLIP's recall and Winoground with the negatives' binding.
2. Learnable temperature + negatives (the InfoNCE negatives row still has the fixed temperature).
3. Report GroupMatch (Zhu et al. ICLR 2026) next to standard Winoground.
4. Larger changes wait for the scene-graph step (structure on both sides) and step 11 (soft composition).

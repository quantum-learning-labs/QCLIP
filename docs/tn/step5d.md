# Step 5d: negatives under a learned scale, rebalanced and down-weighted negatives, GroupMatch (report, 18 Nov 2026)

## 1. Answer
- **Learned temperature + hard negatives keeps nearly all the recall.** InfoNCE with a learnable logit scale and the step-5c
  negatives gives rsum 325.1 (325.8 / 324.5), against 329.9 without negatives and 272.2 for the same negatives at the fixed
  temperature 0.07. Binding is kept: ARO-Att 0.725, ARO-Rel 0.576, swap_full 0.996, SC++ replace-relation 0.448, SVO 0.845.
  The steep recall cost of negatives in step 5c was mostly a fixed-temperature effect, like the hubness.
- **Each benchmark follows its own kind of negative.** Shifting the negatives toward relation replacement (v2: relation 74%,
  noun 20%, modifier 6%) raises SC++ replace-relation (0.463, best of any TN) but lowers ARO-Rel (0.564), ARO-Att (0.711) and
  swap_full (0.966). ARO VG-Relation and swap_full negatives swap the two nouns' roles, so noun swaps are what they reward;
  ARO-Att follows the modifier share.
- **The negative weight trades recall for binding.** SigLIP + v2 negatives at weight 0.3: rsum 329.0, Winoground text 0.178,
  ARO-Att back to 0.623.
- **GroupMatch** (Zhu et al., ICLR 2026; for a 2x2 Winoground item correct iff s00 + s11 > s01 + s10, so text = image =
  group, chance 0.5): every step-5b/5c/5d TN scores 0.48-0.55 (n = 390). The TN does not order Winoground pairs above
  chance under any objective tried. CSLS cannot change GroupMatch (the offsets cancel).
- **Scope (user's point):** hard negatives, SigLIP and the learned scale are model-agnostic; they would help the MLP tree
  baseline too. The ~320 bar was set by an InfoNCE MLP tree, so the TN's 335 (SigLIP) is not yet a like-for-like win. The
  matched MLP control (section 6) decides what is TN-specific.

## 2. Setup
Base as step 5c (CP128 + R1, EMA 0.999, constant LR 1e-3, batch 512, 150 epochs, best.pt by val MRR), 2 seeds each.
Code tn-step5d: e4639eb + 691f334 (neg_weight was first patched into BaseInfoNCELoss; fixed before any training run).
- `neg_weight` w: InfoNCE adds log w to the negative logits (w copies in the softmax); SigLIP scales the negative pair
  terms by w. At w = 1 both skip the new code; checked bit-identical steps.csv losses against 999605d (same seed, same pool
  GPU) and unit-checked both forms.
- v2 negatives: tools/tn/s5_make_negs.py policy prefer_rel (relation > modifier > noun when available) ->
  data/mscoco/processed/tn/train_negs_v2.pkl (new file, md5 b9868757...): relation 416,657, noun 113,390, modifier 32,706,
  none 1,697 of 564,450 captions (v1: noun 57%, relation 34%, modifier 9%).
- Pool node barnacle segfaults at random in pickle I/O and GC (bad hardware); two v2 files it wrote are kept as *.bad-*
  and unused.

## 3. Results (test, mean of 2 seeds)
| model | seeds | rsum | ARO-Rel | ARO-Att | SVO | swap_full | SC++ repl-obj | SC++ repl-att | SC++ repl-rel | SC++ swap-obj | SC++ swap-att | Wino text | Wino image |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| InfoNCE, learnable temperature (5c) | 2 | 329.9 | 0.517 | 0.568 | 0.844 | 0.933 | 0.759 | 0.516 | 0.425 | 0.357 | 0.375 | 0.192 | 0.068 |
| SigLIP (5c) | 2 | 335.2 | 0.522 | 0.569 | 0.849 | 0.924 | 0.747 | 0.519 | 0.413 | 0.371 | 0.381 | 0.206 | 0.092 |
| hard negatives, fixed T (5c) | 2 | 272.2 | 0.585 | 0.803 | 0.805 | 0.995 | 0.714 | 0.449 | 0.420 | 0.416 | 0.470 | 0.106 | 0.082 |
| SigLIP + hard negatives (5c) | 2 | 318.9 | 0.575 | 0.755 | 0.841 | 0.995 | 0.705 | 0.469 | 0.443 | 0.410 | 0.447 | 0.153 | 0.094 |
| **learnable T + negatives** | 2 | 325.1 | 0.576 | 0.725 | 0.845 | 0.996 | 0.747 | 0.503 | 0.448 | 0.371 | 0.425 | 0.169 | 0.077 |
| SigLIP + v2 negatives | 2 | 324.5 | 0.564 | 0.711 | 0.845 | 0.966 | 0.712 | 0.480 | 0.463 | 0.385 | 0.382 | 0.151 | 0.081 |
| SigLIP + v2 negatives, w 0.3 | 2 | 329.0 | 0.554 | 0.623 | 0.843 | 0.982 | 0.729 | 0.494 | 0.462 | 0.348 | 0.391 | 0.178 | 0.085 |

Per seed rsum: learnable T + negatives 325.8 / 324.5; SigLIP + v2 325.4 / 323.5; w 0.3 329.5 / 328.4.

## 4. CSLS and GroupMatch (k = 10, alpha = 0.5, COCO-val bank; results/tn/step5/audit/wino_hub_5d_*.json)
| model | raw rsum | CSLS | gain | Wino text / image (CSLS) | GroupMatch |
|---|---|---|---|---|---|
| learnable T + negatives | 325.1 | 343.2 | +18.0 | 0.177 / 0.077 | 0.483 |
| SigLIP + v2 negatives | 324.5 | 337.5 | +13.1 | 0.164 / 0.092 | 0.492 |
| SigLIP + v2 negatives, w 0.3 | 329.0 | 342.1 | +13.1 | 0.190 / 0.091 | 0.508 |
GroupMatch of earlier runs (wino_hub_best.json, wino_hub_5c.json): EMA 0.50-0.52, SigLIP 0.53-0.55, negatives 0.49-0.54,
step-4c o16v3 0.546, BoW 0.267 (identical bags of words tie, and ties count as wrong).

## 5. What it says
- With the scale learned, negatives cost about 5 rsum instead of about 57: most of the step-5c recall cost of negatives came
  from the fixed temperature. The step-5 recipe for binding is learnable temperature + v1 negatives.
- The benchmarks measure different swaps, and the negative mix decides which one improves (noun swaps -> ARO-Rel /
  swap_full, modifier swaps -> ARO-Att, relation replacement -> SC++ replace-relation). This is a training-data effect and
  model-agnostic, as the user noted; it says nothing yet about the TN as an architecture.
- Winoground stays at chance in GroupMatch for every TN: the readout of a pooled CLIP image vector against one sentence
  vector does not order these pairs, whatever the loss. That points to the image side (no structure), the target of the
  scene-graph step.
- Thesis view: everything here changes the loss or the data, not the (graph, generator, readout) triple; consistent with the
  first-year conclusion that the readout / objective, not the circuit, drives these gaps.

## 6. Not done: matched MLP control
The repo's MLP tree (`model_type: mlp`, one WordMLP per word-type over the same wiring, 234M params) trains on the frozen CLIP
vectors and benchmarks end to end (smoke on pintail). It runs a Python loop over tree nodes: 1.84 s/step (InfoNCE), 3.26 s/step
with negatives, i.e. ~17 h / ~30 h for 150 epochs on a pool GPU. Waiting for the user's choice: batch the MLP forward per tree
level (equal outputs proven) and run 150 epochs; or 50 epochs; or > 24 h runs. Recipes: InfoNCE fixed T, SigLIP,
SigLIP or learnable T + negatives; table of Delta(objective) for TN vs MLP.

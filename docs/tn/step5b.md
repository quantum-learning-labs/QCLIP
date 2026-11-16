# Step 5b: follow-ups on CP128 (report, 16 Nov 2026)

## 1. Answer
The step-5 recipe is **CP rank 128 + R1 codomain maps + EMA 0.999 of the weights** (constant LR 1e-3, batch 512, 150 epochs,
best.pt by val MRR), read out with **CSLS** at test time. Test rsum 288.1 raw (290.1 / 286.0), **318.0 with CSLS**
(319.9 / 316.1), against 275.1 for CP128 + R1 and the ~320 bar of the MLP tree model (same tree + frozen CLIP B/32 image
vector, contrastive). Nothing else in the screen moves recall by more than +4, and no change moves the compositional scores
(ARO, SC++, Winoground) beyond seed noise except E1, which pays for it in recall.
- EMA works mostly as the annealing the constant-LR runs never had: a cosine schedule alone gives +8, and cosine on top of EMA
  is worse than EMA alone (283.7). EMA + cosine + batch 1024 is worse again (280.0): half the updates, annealed LR.
- CSLS (k = 10, alpha = 0.5, COCO-val caption/image bank, no test data) adds +29 to +31 rsum to every model.
- Lifting (rules R2-R6, full lifting) is neutral on every benchmark. Keep R1 (free, one codomain map per atomic type);
  full lifting matters for the graph family (lower treewidth), not for scores.
- E1 (own CP tensors for closed-class relation words) is the only change that moves relation scores (ARO-Rel +0.024,
  SC++ replace-relation +0.03) and costs 36-53 rsum, more with more own tensors.

## 2. Setup
Base: CP128 + R1 from step 5 (decoupled words, shared argument cores by leg/side/distance, centred base-point words,
scale_free, no-bias head, levelwise engine). Test sets and metrics as in every step (COCO 5k rsum, ARO, SVO-probes,
SugarCREPE++ "both" accuracy, Winoground). 2 seeds per configuration. Code: branch tn-step5 (commits 0494e53 screen,
37c9435 E1, 2d596da wino_hub, 68df61f treewidth, bbe3129 NaN fix). Runs on the UCL pool (3090 Ti) and the HPC cluster.

## 3. Results (test, mean of 2 seeds)
| model | seeds | rsum | ARO-Rel | ARO-Att | SVO | swap_full | SC++ repl-obj | SC++ repl-att | SC++ repl-rel | SC++ swap-obj | SC++ swap-att | Wino text | Wino image |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base CP128 + R1 | 2 | 275.1 | 0.521 | 0.549 | 0.795 | 0.945 | 0.739 | 0.447 | 0.368 | 0.348 | 0.312 | 0.138 | 0.074 |
| EMA 0.999 | 2 | 288.1 | 0.520 | 0.570 | 0.799 | 0.950 | 0.772 | 0.466 | 0.381 | 0.357 | 0.315 | 0.144 | 0.067 |
| cosine LR | 2 | 283.1 | 0.527 | 0.549 | 0.804 | 0.954 | 0.745 | 0.427 | 0.379 | 0.334 | 0.298 | 0.142 | 0.072 |
| EMA + cosine | 2 | 283.7 | 0.527 | 0.551 | 0.802 | 0.955 | 0.743 | 0.430 | 0.378 | 0.328 | 0.293 | 0.140 | 0.065 |
| batch 1024 | 2 | 278.8 | 0.538 | 0.579 | 0.801 | 0.959 | 0.747 | 0.427 | 0.365 | 0.369 | 0.294 | 0.150 | 0.078 |
| EMA + cosine + batch 1024 | 2 | 280.0 | 0.530 | 0.552 | 0.801 | 0.959 | 0.734 | 0.419 | 0.368 | 0.336 | 0.283 | 0.147 | 0.073 |
| 300 epochs | 2 | 278.0 | 0.524 | 0.557 | 0.795 | 0.931 | 0.751 | 0.442 | 0.372 | 0.330 | 0.317 | 0.131 | 0.076 |
| CLIP word init | 2 | 277.4 | 0.526 | 0.571 | 0.804 | 0.937 | 0.746 | 0.455 | 0.365 | 0.336 | 0.300 | 0.150 | 0.077 |
| weight decay 0.01 | 2 | 274.0 | 0.528 | 0.579 | 0.802 | 0.957 | 0.710 | 0.411 | 0.365 | 0.350 | 0.282 | 0.129 | 0.073 |
| weight decay 0.1 | 2 | 269.5 | 0.526 | 0.568 | 0.797 | 0.945 | 0.719 | 0.417 | 0.369 | 0.359 | 0.293 | 0.127 | 0.077 |
| word dropout 0.1 | 2 | 270.2 | 0.515 | 0.556 | 0.799 | 0.943 | 0.746 | 0.430 | 0.360 | 0.328 | 0.272 | 0.136 | 0.079 |
| R1 + R2 | 2 | 275.4 | 0.531 | 0.556 | 0.792 | 0.957 | 0.740 | 0.451 | 0.353 | 0.316 | 0.301 | 0.150 | 0.062 |
| R1 + R3 | 2 | 275.4 | 0.520 | 0.557 | 0.792 | 0.963 | 0.751 | 0.427 | 0.363 | 0.348 | 0.309 | 0.144 | 0.086 |
| R1 + R4 | 2 | 273.1 | 0.529 | 0.558 | 0.791 | 0.943 | 0.741 | 0.445 | 0.371 | 0.330 | 0.289 | 0.136 | 0.081 |
| R1 + R5 | 2 | 274.0 | 0.526 | 0.583 | 0.806 | 0.946 | 0.748 | 0.440 | 0.377 | 0.309 | 0.278 | 0.159 | 0.064 |
| R1 + R6 | 2 | 276.1 | 0.523 | 0.565 | 0.809 | 0.951 | 0.753 | 0.420 | 0.366 | 0.309 | 0.293 | 0.156 | 0.063 |
| full lifting, 300 ep | 2 | 274.6 | 0.526 | 0.574 | 0.795 | 0.934 | 0.730 | 0.453 | 0.358 | 0.357 | 0.309 | 0.135 | 0.086 |
| E1 own relation tensors (R1) | 2 | 222.0 | 0.545 | 0.544 | 0.781 | 0.963 | 0.614 | 0.339 | 0.399 | 0.309 | 0.275 | 0.147 | 0.096 |
| E1 own relation tensors (full lifting) | 2 | 239.1 | 0.543 | 0.553 | 0.795 | 0.966 | 0.641 | 0.349 | 0.387 | 0.332 | 0.286 | 0.149 | 0.079 |

Chance: ARO, SVO 0.5; SC++ 0.333; Winoground text/image 0.25. CLIP B/32's own text encoder on the same image vectors: 359.6.

## 4. Test-time hubness correction (tools/tn/wino_hub.py; results/tn/step5/audit/wino_hub*.json)
| model | raw rsum | CSLS k10 a0.5 | CSLS k10 a1.0 | centring |
|---|---|---|---|---|
| bag-of-words | 305.3 | 338.4 | 328.5 | 294.4 |
| CP128 (no lifting), 2 seeds | 275.0 | 304.2 | | 265.9 |
| CP128 + R1 | 274.8 | 305.7 | | 265.3 |
| EMA, 2 seeds | 288.1 | **318.0** | 308.7 | |
| cosine, 2 seeds | 283.1 | 313.2 | 302.3 | |
| EMA + cosine + batch 1024 | 280.0 | 310.7 | 299.4 | |
Winoground is below chance under every correction (CP128 text 0.14-0.19, image 0.05-0.12; BoW image 0.015): the deficit is
the model's, not a hubness artefact. CSLS is known (Conneau et al. 2018; QB-Norm, Bogolin et al. 2022); it is reported as a
separate readout column, and a fair comparison to the MLP baseline needs it applied there too.

## 5. What each follow-up showed
- **Training dynamics.** EMA +13.0, cosine +8.0, batch 1024 +3.7, 300 epochs +2.9, CLIP word init +2.3 (stand-in only).
  Weight decay 0.01 neutral, 0.1 -5.6; word dropout 0.1 -4.9. Best epochs are late (130-149) in every 150-epoch run, so
  early stopping would save little; annealing is what was missing.
- **Word dropout divergence (fixed, bbe3129).** A caption whose words are all dropped gives a centred difference of exactly
  0; normalize's eps gradient (1e33) overflowed to NaN. Now a zero row gets denominator 1 (identical output elsewhere).
- **Lifting ablation.** R1 + {R2, R3, R4, R5, R6}: rsum 273.1-276.1 against 275.1; R5 raises ARO-Att to 0.583 (both seeds).
  Full lifting needs 300 epochs to match (270.3 at 150).
- **E1 own relation tensors.** 124 (word, type) pairs, +8M parameters: 222.0; 82 pairs with full lifting: 239.1. Gains:
  ARO-Rel 0.545, SC++ replace-relation 0.399, swap_full 0.963-0.966. Losses: SC++ replace-object 0.739 -> 0.614,
  replace-attribute 0.447 -> 0.339. Validation MRR plateaus from epoch 30, so it is not undertraining. The own tensors are
  smaller than the shared ones (median norm 2.3 vs 4.5), so scale is not the cause; the candidate is routing: noun
  information passes through relation tensors trained on far fewer uses. This confirms the step-5 diagnosis (the shared
  relation tensor lets training ignore the relation word) and shows that capacity alone is an expensive fix.
- **Treewidth (tools/tn/s5_treewidth.py).** Index-graph treewidth (contraction cost) is 2-4 for most unlifted captions; full
  lifting gives 2 for 88% of captions (max 14 over all): lifting lowers contraction cost despite the cycles it adds.

## 6. Not done / held back
- CSLS is not applied inside training; GroupMatch (Zhu et al. ICLR 2026) not yet reported.
- SigLIP loss and diagram-level hard negatives: planned, not started.
- 2 seeds per configuration; the pool was contended for most of the screen (several runs moved between cluster and pool).

## 7. Next
1. SigLIP loss and hard negatives built from the diagrams (swap nouns across relation legs, replace prepositions, swap
   attributes; same wiring, no re-parse), on the EMA recipe. Goal: beat ~320 without CSLS and lift the relation subsets.
2. Objectives that need structure on both sides (graph matching, per-node grounding, joint likelihood) wait for the
   scene-graph step; soft composition (step 11) is designed (Future Step page) and not started.

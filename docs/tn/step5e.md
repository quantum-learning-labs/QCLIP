# Step 5e: width, affine composition, and where the generalisation gap comes from (report, 25 Nov 2026)

## 1. Answer
- **Bag-of-words with the same word vectors reaches rsum 370.5 (2 seeds) under the current recipe.** The TN gets 325.2.
  Composition costs about 45 rsum and buys almost all the binding: ARO-Att 0.725 vs 0.263, ARO-Rel 0.576 vs 0.273,
  swap_full 0.996 vs 0.21, Winoground text 0.169 vs 0.06. Additive composition being a strong baseline against tensor-based
  composition is known (Milajevs et al., EMNLP 2014). What is new here is where the cost comes from.
- **The train-test gap belongs to the lexicon, not to composition.**
  - BoW: train 451 / test 370 (gap 81). TN: train 412 / test 326 (gap 86). The gaps are about the same.
  - In both models the gap concentrates on captions with rare words (t2i R@10, words seen 1-9 times: TN 0.96 train /
    0.60 test, BoW 0.94 / 0.68; unseen word: TN 0.49, BoW 0.60).
  - The per-word free vectors memorise images through rare words. Step 6 (token sub-networks for words) is the
    architectural answer to this.
- **The composition cost shows on training images too** (TN 412 vs BoW 451), so it is not a generalisation effect: it is
  expressivity or trainability. The TN is also much slower to fit:
  - BoW reaches 366 test by epoch 20 and plateaus by epoch 40-50 (best 45/52).
  - The TN is still rising at epoch 150 (test 312 at epoch 60, 325 at epoch 140).
- **Composition transfers to unseen diagram shapes in absolute terms.** TN test R@10 is flat across shape frequency (0.65-0.67,
  2 seeds + SigLIP), with unseen shapes covering 14% of test captions.
  - But relative to BoW the TN loses more on rare shapes: TN − BoW is −0.10 for unseen shapes and −0.04 for shapes seen 1000+
    times.
  - So the shared generators are less well identified on rare structures (thesis requirement I).
- **Width 256: +4 rsum** (328.1 / 330.1 = 329.1, 2 seeds). Capacity is not the limit.
- **Pure affine-multilinear composition (homogeneous coordinates) cannot be trained at caption depth: negative result.**
  - The constant slot sets scale_free's max-|entry| divisor, so the multilinear part shrinks by about (a·x)^k per level.
  - The 512-d output (slot dropped) ends up near 1e-20, and unit-normalising it makes the step-0 gradients 2e15-8e21, which
    overflow when squared.
  - RNTN and Tree-LSTM bound this with tanh. A per-node normalised variant would train, but it is no longer a TN, so it was
    not pursued.

## 2. Setup
Base = step-5d recipe: CP128 + R1, learnable temperature + v1 diagram negatives, EMA 0.999, constant LR 1e-3, batch 512,
150 epochs, best.pt by val MRR. Code tn-step5e: 51bcef0 (text.affine), 783c288 (scale_free leaves all-zero intermediates
unscaled; bit-identical plain losses vs 51bcef0).
- **BoW:** `text.arch: bow`. It uses the mean of the same 128-d word vectors (word_base + per-word), bow_proj 128→512, and the
  same head; the type tensors are unused. All other flags are as the TN runs. It trains at 3.6 s/epoch against 3+ min/epoch
  for the TN.
- **Tools:**
  - `tools/tn/s5e_train_gap.py`: rsum on 5k train images / val / 5k test, same encode path.
  - `tools/tn/s5e_gap_breakdown.py`: per-caption tie-neutral t2i R@k with lexical and structural keys from the training
    compile. Shape = einsum + non-word symbols, word identity ignored. Its overall R@1/5/10 reproduces benchmark_best_final.json
    exactly.
  - `tools/tn/s5e_grad_probe.py`: step-0 non-finite and largest gradients per bank.
- Results: results/tn/step5/audit/gap_curve/*.json and gap_breakdown/*.json (worktree, untracked).

## 3. Results (test)
| model | seeds | rsum | train5k rsum | ARO-Att | ARO-Rel | SVO | swap_full | SC++ repl-obj / att / rel | SC++ swap-obj / att | Wino text / image |
|---|---|---|---|---|---|---|---|---|---|---|
| TN, learnable T + negatives (5d) | 2 | 325.2 | 412.3 (s0) | 0.725 | 0.576 | 0.845 | 0.996 | 0.747 / 0.503 / 0.448 | 0.371 / 0.425 | 0.169 / 0.077 |
| TN SigLIP (5c) | 1 (s0) | 333.9 | 422.0 | | | | | | | |
| TN width 256 | 2 | 329.1 | | 0.747 | 0.569 | 0.844 | 0.995 | 0.738 / 0.514 / 0.444 | 0.420 / 0.440 | 0.156 / 0.086 |
| **BoW, same words + recipe** | 2 | **370.5** | 451.2 (s0) | 0.263 | 0.273 | 0.846 | 0.21 | 0.865 / 0.690 / 0.594 | 0.150 / 0.246 | 0.060 / 0.015 |
| DisCoViz Tree-LSTM (ref.) | | 367 | | 0.679 | 0.591 | | | | | |
| DisCoViz RNTN (ref.) | | 337 | | 0.678 | 0.578 | | | | | |

ARO/swap below 0.5 for BoW: order-blind models tie on swapped captions, and ties count as wrong.

**Curves (s0, rsum train5k / test):**

| epoch | 10 | 20 | 40 | 60 | 100 | 140 |
|---|---|---|---|---|---|---|
| TN | 268 / 242 | 312 / 277 | 351 / 302 | 371 / 313 | 396 / 321 | 411 / 325 |
| BoW | 386 / 354 | 412 / 366 | 429 / 370 | 438 / 371 | 447 / 370 | 451 / 370 |

**Per caption, test t2i R@10 (TN = ltneg s0, BoW = bow s0; train in brackets):**

| bucket | n | TN | BoW | TN − BoW |
|---|---|---|---|---|
| rarest word unseen | 562 | 0.489 | 0.596 | −0.11 |
| rarest word seen 1-9 | 1463 | 0.600 (0.960) | 0.684 (0.936) | −0.08 |
| 10-99 | 4884 | 0.678 (0.919) | 0.757 (0.938) | −0.08 |
| 100-999 | 10087 | 0.695 (0.838) | 0.768 (0.870) | −0.07 |
| 1000+ | 7913 | 0.643 (0.732) | 0.679 (0.749) | −0.04 |
| shape unseen in train | 3553 | 0.660 | 0.759 | −0.10 |
| shape seen 1-9 | 3635 | 0.669 | 0.739 | −0.07 |
| shape seen 100-999 | 7735 | 0.666 | 0.718 | −0.05 |
| shape seen 1000+ | 3978 | 0.657 | 0.695 | −0.04 |

The same pattern holds for TN s1 and SigLIP s0, and for BoW s1.

## 4. Affine composition (negative result)
- **Model:** `text.affine` contracts every argument as [x; 1]. Each CP component is then prod_k (a_k·x_k + c_k), which holds
  the multilinear term plus every lower-order term down to a constant. Masks pin the slot entries. Zero bias equals the plain
  model, checked to float rounding.
- **Training:** runs collapsed to chance (t2i MRR 0.002 for 12 epochs) with gradient spikes up to 2.4e9.
- **Probe:** no all-zero intermediates, 16 below 1e-20; grad max 2e15 (argument factors) and 8e21 (m:->out); the plain model's
  maximum is 59. Clipping zeroes the update, and turning scale_free off does not help either.
- **Mechanism:** the slot keeps magnitude 1 while the real part shrinks by about |a·x| (~0.1-0.3) per argument and level.
- **Thesis-table note:** adding homogeneous coordinates to the same graph and generator family makes deep trees untrainable
  without scale control.

## 5. Next
Step 5 closes here. Recipe: CP128 + R1 + EMA 0.999, learnable temperature (or SigLIP) + diagram negatives.
- **Step 6 (token sub-networks):** targets the lexical gap directly (rare and unseen words), and it is the gap that BoW shares.
- **The composition cost (−45 rsum, present on training images, slow fit, larger on rare shapes):** an expressivity and
  trainability question about multiplicative composition. Steps 6/7 should be read with it in mind.

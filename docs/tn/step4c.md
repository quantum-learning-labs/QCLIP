# Step 4c: composition that keeps recall

Branch `tn-step4c` (off `tn-step4`), worktree `comp_bridge/qclip_wt/step4c`, commits 8256c6c, a1e1778, 92841a0. Plan:
`~/.claude/plans/tn-step4c-composition.md`. Full per-benchmark tables (mean ± std, chance, z): `step4c_tables.md`
(results/tn/step4c/final). Diagnostics: results/tn/step4c/diag. All runs: the step-4 recommended setup (dec_t3, scale_free,
init_orth, no-bias head, LR 1e-3 constant, 150 epochs, best.pt by val mrr), CLIP B/32 image embeddings frozen, InfoNCE, batch 512.
Recall is a fraction: R@10 0.35 = the correct item is in the top 10 for 35% of queries (5k images, ~25k captions).

## Result
- **Goal** (step 4 found that the TN loses recall to an order-blind bag-of-words model, 0.35 vs 0.76 i2t R@10, while keeping word
  order): close the gap without giving up composition, using architecture/parametrisation only (no new objectives).
- **Best model: shared output core with output bond 16 + centred base-point words (o16v3)**: i2t/t2i R@10 **0.415 / 0.304**
  (step 4: 0.350 / 0.245; +19% / +24%), SVO swap_full 0.976, ARO-Rel 0.548, SVO std 0.748, all at least as good as step 4.
  16.2M text params (step 4: 18.0M), 5.4 GB peak. It matches the step-4b root-bond-32 model (0.440 / 0.305) at 1/8.6 of its params
  and 28% of its memory, and it has **no early plateau** (val i2t mrr 0.17 by epoch 15; every other TN variant sits on the
  step-4 plateau for 45-130 epochs). Cost: 2.9x slower per step than step 4 (a second, de-duplicated contraction pass).
- **The target (i2t R@10 >= 0.6 with swap_full >= 0.9) is not reached.** The gap to bag-of-words (0.759) is now internal to the
  type tensors (section 4), which is the step-5 question (decomposition / internal rank).
- **A lexical path beside the TN kills composition**: hybrid TN + bag-of-words (learned or fixed mix) reaches bag-of-words recall
  (0.75) but swap_full falls to chance (0.49-0.50) and ARO-Rel to 0.50: the TN path stops carrying word order.
- **Base-point word vectors** (word = shared base + small offset, V2) are undone by training (the base shrinks below the offsets);
  centring (V3, subtract the same network with every word = base) keeps the base and adds ~+5-9% recall.

## 1. Candidates (each one config key, default off; defaults bit-identical to tn-step4: 444/444 step losses)
| key | what it does |
|---|---|
| `text.compose: hybrid` (V1) | sentence = unit(TN) + mix * unit(P mean(word vectors)), mix learned (init 1); `text.mix_fixed: 1.0` (R2) holds it |
| `text.word_base: alpha` (V2) | word vector = b + d_w, b one shared learned vector, d_w per word (init scaled by alpha); to first order the TN is additive in the words |
| `text.centre: true` (V3) | V2 minus the same network with every word = b (the caption's "empty structure" term), computed exactly under scale_free by tracking per-sentence log scales; base pass de-duplicated per (structure, type rows) |
| `text.share_out: true` | the MPS core holding the 512-d output leg is ONE tensor shared by all 138 root types (was one per root type) |
| `text.out_bond_dim: k` | only the two bonds next to that core widened to k (span per caption <= k^2 instead of 64); other bonds stay 8 |
Checks: tools/tn/s4c_test.py (centred output = fp64 reference, cos 1.000000 on 1024 captions; d = 0 gives the base-only pass;
gradients reach every new parameter), tools/tn/s4c_decomp_test.py (decomposer defaults = old decomposer on 500 random types).

## 2. Results (2 seeds each, 150 epochs; step 4: 5 seeds; step 1: 3 seeds at 3 epochs)
| | i2t R@10 | t2i R@10 | i2t R@1 | SVO swap_full | ARO-Rel | ARO-Att | SVO std | SC++ repl-obj | text params |
|---|---|---|---|---|---|---|---|---|---|
| step 1 reference | 0.041±0.004 | 0.009±0.001 | 0.007±0.001 | 0.497±0.011 | 0.494±0.007 | 0.488±0.014 | 0.509±0.019 | 0.375±0.023 | 537.9M |
| step 4 (recommended) | 0.350±0.005 | 0.245±0.006 | 0.071±0.002 | 0.966±0.013 | 0.544±0.004 | 0.525±0.012 | 0.725±0.005 | 0.476±0.010 | 18.0M |
| bag-of-words (1 seed) | 0.759 | 0.590 | 0.367 | 0.205 | 0.256 | 0.250 | 0.789 | 0.858 | 18.1M |
| V1 hybrid | 0.756±0.001 | 0.585±0.001 | 0.361±0.002 | 0.494±0.018 | 0.503±0.001 | 0.499±0.002 | 0.780±0.007 | 0.856±0.003 | 18.1M |
| R2 hybrid, mix fixed 1 | 0.749±0.000 | 0.575±0.001 | 0.352±0.004 | 0.504±0.009 | 0.499±0.008 | 0.511±0.003 | 0.779±0.005 | 0.846±0.005 | 18.1M |
| V2 alpha 0.1 | 0.344±0.002 | 0.249±0.000 | 0.068±0.001 | 0.969±0.012 | 0.543±0.003 | 0.538±0.009 | 0.722±0.004 | 0.501±0.011 | 18.0M |
| V2 alpha 0.3 | 0.345±0.006 | 0.255±0.000 | 0.069±0.004 | 0.936±0.041 | 0.544±0.004 | 0.539±0.003 | 0.733±0.003 | 0.509±0.000 | 18.0M |
| V3 centred | 0.366±0.006 | 0.268±0.008 | 0.075±0.004 | 0.962±0.013 | 0.548±0.000 | 0.537±0.004 | 0.736±0.002 | 0.505±0.026 | 18.0M |
| o8 shared out core | 0.336±0.004 | 0.239±0.001 | 0.067±0.003 | 0.938±0.004 | 0.544±0.006 | 0.522±0.007 | 0.716±0.006 | 0.527±0.000 | 14.6M |
| o16 (+ out bond 16) | 0.387±0.002 | 0.278±0.002 | 0.088±0.002 | 0.974±0.004 | 0.547±0.003 | 0.525±0.018 | 0.753±0.015 | 0.494±0.011 | 16.2M |
| o32 (+ out bond 32) | 0.406±0.022 | 0.280±0.026 | 0.097±0.013 | 0.972±0.006 | 0.550±0.001 | 0.514±0.013 | 0.744±0.007 | 0.468±0.009 | 19.8M |
| **o16v3 (o16 + V3)** | **0.415±0.004** | **0.304±0.000** | 0.095±0.002 | **0.976±0.007** | 0.548±0.006 | 0.525±0.006 | 0.748±0.006 | 0.532±0.005 | 16.2M |
| step-4b root bond 32 (c11*) | 0.440±0.002 | 0.305±0.001 | 0.106±0.007 | 0.967±0.005 | 0.545±0.005 | 0.533±0.001 | 0.756±0.001 | 0.463±0.027 | 139.2M |
*c11 = step-4 constraint C11 (grad clip) with the old affine head; listed for its output capacity only.
SugarCREPE++ swap/replace-attribute/relation and Winoground stay below chance for every TN variant (step4c_tables.md), as in step 4.
o32's spread comes from a late escape: seed 1 left the plateau only at epoch ~125 and was still rising at 150 (val i2t mrr 0.183,
seed 0: 0.214); o32 is under-trained at 150 epochs, o16v3 is converged (flat val mrr ~0.20 from epoch 45).

## 3. Why the first three candidates fail (learned parameters)
- V1/R2: the learned mix grows 1 -> 5.9 (V1); with it fixed at 1 (R2) the outcome is the same: once a full-rank lexical path exists,
  the gradient flows there and the TN path loses its word-order information (swap at chance in all 4 runs).
- V2: base norm 1.03 -> 0.18-0.20 while the offsets grow (alpha 0.1: 0.10 -> 0.22; 0.3: 0.30 -> 0.36): |b|/|d| goes from 10 (or 3.4)
  to < 1, i.e. training returns to the multiplicative regime; V2 = step 4 on every benchmark.
- V3 keeps |b| 0.44 vs median |d| 0.39 and lowers word-swap sensitivity below bag-of-words (table below), yet recall rose only
  +5-9%: the output span was still capped at 64.

## 4. Diagnostics (seed 0 unless noted; tools/tn/s4_{influence,embed,paraphrase}.py)
| | word-swap influence (1-cos) | words ignored | top word's share | text eff. rank | t2i R@10 by length <=8 / 9-10 / 11-12 / 13+ | same-image cos same / diff root | max k-occurrence |
|---|---|---|---|---|---|---|---|
| step 4 | 0.540 | 7% | 33% | 22 | 0.26 / 0.26 / 0.25 / 0.21 | 0.67 / 0.56 | 705 |
| bag-of-words (step 4b) | 0.15 | - | - | 51 | 0.57 / - / - / 0.65 | 0.72 / 0.70 | 408 |
| V1 hybrid | 0.133 | 0% | 23% | 50 | 0.56 / 0.57 / 0.60 / 0.64 | 0.72 / 0.70 | 388 |
| V2 alpha 0.3 | 0.124 | 21% | 43% | 22 | 0.27 / 0.26 / 0.26 / 0.21 | 0.71 / 0.59 | 665 |
| V3 | 0.082 | 27% | 46% | 22 | 0.27 / 0.27 / 0.26 / 0.23 | 0.72 / 0.60 | 670 |
| o16 | 0.547 | 5% | 32% | 25 | 0.29 / 0.29 / 0.27 / 0.23 | 0.66 / 0.57 | 578 |
| o32 | 0.560 | 5% | 31% | 30 | 0.30 / 0.31 / 0.30 / 0.26 | 0.64 / 0.54 | 499 |
| o16v3 (s0; s1 equal within 0.01) | 0.087 | 26% | 43% | 24 | 0.31 / 0.31 / 0.31 / 0.28 | 0.71 / 0.62 | 502 |
Reading:
- Lifting the output cap (64 -> 256 / 512) raises recall and flattens the length curve, but the text embeddings still use only
  24-30 effective dimensions (bag-of-words 51, CLIP text encoder 68, images 65). The output leg is no longer the binding limit.
- Each argument (a 128-d n/s/p wire) enters a type tensor through 8-dim MPS bonds before it reaches the output core: every internal
  bond is still 8. That internal rank is the remaining bottleneck, and it is what step 5 (decomposition of the type tensors) varies.
- Centring makes a caption depend on a few dominant words (26% of words below 0.01 influence, top word 43%), not on all words
  evenly; it removes the per-root-type offset (same-image agreement across root types 0.56 -> 0.62) and the early plateau.
- 35 of 138 root types have the output leg at a chain end (span <= 8, ~2% of captions, e.g. t:s@n@n@out rank 8); share_out
  gives them the same core shape as the rest of their position class.

## 5. Efficiency (controlled: one L40S, 2 epochs each, same seed; `tn_s4c_prof_*`)
| | step 4 | o8 | o16 | o32 | o16v3 | V3 + root bond 32 (not trained) |
|---|---|---|---|---|---|---|
| text params | 18.04M | 14.56M | 16.23M | 19.75M | 16.23M | 139.25M |
| peak training memory | 4.33 GB | 4.29 GB | 5.44 GB | 13.62 GB | 5.44 GB | 19.29 GB |
| s/step (epoch 2) | 0.205 | 0.223 (1.09x) | 0.278 (1.36x) | 0.283 (1.38x) | 0.590 (2.87x) | 0.753 (3.67x) |
V3 alone: 2.1x step time (cluster check job). The V3 + root-bond-32 run was replaced by o16v3 before it started (user: optimise
with parameter sharing first); its cost row is from the profile only.

## Caveats
- 2 seeds per variant; GPU types mixed across runs (speed only from the controlled profile). Bag-of-words has 1 seed.
- o32 is under-trained at 150 epochs (late escape); o32 + V3 was not run.
- The influence/paraphrase tools use 2,000 test captions (influence) and all coco5k captions (embed, paraphrase).

## Input for step 5
- Use o16v3 as the step-5 base (or o32 + V3 if a longer budget is allowed): shared output core, output bond >= 16, centred words.
- The binding limit is now the internal rank of the type tensors (all non-output bonds 8; eff. rank 24-30). Step 5 should vary that
  rank through the decomposition (MPS bond, CP rank, Tucker ranks, tensor ring) at matched parameter budgets, with and without
  init_orth, and report rank vs params vs accuracy; parameter sharing of cores across complex types is the lever that keeps
  higher ranks affordable (as share_out did for the output core).

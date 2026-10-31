# Step 4: regularisation and constraints (and why runs jump, and why recall is low)

Branch `tn-step4` (off `tn-step3`), worktree `comp_bridge/qclip_wt/step4`. Plans: `~/.claude/plans/tn-step4-regularisation.md`,
`~/.claude/plans/tn-step4b-jump-and-recall.md`. Full per-benchmark tables (mean ± std, chance, z): `step4_tables.md`.
Base model: decoupled `dec_t3` (step 3: one 128-d vector per word, one shared tensor per complex type, MPS bond 8 on type tensors
of order >= 3), CLIP B/32 image embeddings frozen, InfoNCE at temperature 0.07, LR 1e-3, batch 512.

## Result
- **Cause of the step-3 divergence found and removed.** Pre-LayerNorm sentence norms spread over ~40 orders of magnitude (the type
  tensors align their gains; their Frobenius norms barely move) and overflow fp32 for some sentences from epoch ~22 (20% of steps
  before NaN). Fix: **scale-free contraction** (`text.scale_free`): every intermediate is divided by its max |entry|. The network is
  multilinear and the head is scale-invariant, so outputs are unchanged up to the LayerNorm eps. None of the 38 scale-free runs
  went NaN (overflow 0% of steps in every screened one), including 30 epochs at LR 3e-3.
- **Norm, spectral, isometric and gauge constraints do not help.** Fixed Frobenius norms (C3) and spectral caps (C4) diverge 2-4x
  *earlier* than no constraint; MPS canonical gauge (C6) diverges at epoch 3-4 and is 10x slower; isometric tensors (C5) and a lower
  type-tensor LR (C2) are stable-ish but learn less composition. Gradient clipping (C1) prevents NaN but clips every step.
  **Orthogonal init (`init_orth`) speeds learning ~2x.**
- **Why some runs "jump" (train loss 5.5 -> 3.2, recall x3) and others stay on a plateau: the text head's shared LayerNorm bias.**
  It learns first, adds the same offset to every caption and collapses the embeddings (effective rank 141 -> ~5 in 10-40 epochs,
  unrelated captions at cosine 0.3, one image in up to 47% of all top-10 lists). The plateau is that collapsed state. With the bias
  removed (`text.head: no_bias` or `plain`) **5 of 5 runs escape by epoch 50-80** and agree closely (test i2t/t2i R@10
  0.350 ± 0.005 / 0.245 ± 0.006), vs 4 of 10 runs with the bias within 300 epochs and 0 of 6 within 150. Whether a biased run escapes is not set by its seed
  (the same seed escaped on one GPU and not on another) and a stuck run is not tipped out by reseeding or 10% noise (0 of 3).
- **Why recall is low.** With the same word vectors, data, loss and frozen image space, a bag-of-words text model reaches i2t/t2i
  R@10 **0.759 / 0.590**, 91% of CLIP's own text encoder (0.834 / 0.661). The best TN reaches 0.44 / 0.30. The data, the frozen image
  side, the temperature and OOV words (2% of captions) are not the limit; the TN composition is: (1) **the output bottleneck**: the
  512-d output leg is a middle core (8, 512, 8) of each root type, so captions of a root type span <= 64 dims; widening the root bond
  to 32 lifts recall +25% (0.35 -> 0.44 i2t R@10) at 7.7x the parameters; (2) **multiplicative composition**: changing any one word
  moves the TN embedding by 1 - cos = 0.55 (bag-of-words 0.15), recall *falls* with caption length (bag-of-words: rises), and
  same-image captions that parse to different root types (70% of pairs) agree less (cos 0.55 vs 0.67 same root; bag-of-words 0.70).
- **Compositional benchmarks**: the TN keeps its word-order advantage (SVO swap_full 0.967, where bag-of-words is at 0.205 because it
  cannot tell swapped captions apart), SVO-probes std 0.73-0.76, ARO VG-Relation 0.54-0.55 (z ~5). SugarCREPE++ swap/replace-attribute
  and Winoground stay below chance for every TN variant (unchanged from step 3).
- **Recommended training setup for step 5:** `text.scale_free: true`, `text.constraint: {init_orth: true}`, `text.head: no_bias`
  (or `plain`), constant LR 1e-3 with best.pt by val mrr; 150 epochs suffice. Cost vs no fix: -10% training throughput, same memory
  and parameters.

## 1. Divergence: diagnosis (phase 0)
- Parameter norms of the step-3 checkpoints: no tensor grows. Type tensors shrink ~2x (median) and a tail becomes spiky
  (sigma_max / sigma_rms of the output-leg unfolding: p99 1.7 at init -> 5-6 before divergence).
- New per-step logging (`tn_stats`: gradient norm per group, pre-LayerNorm norm range; `paramstats.csv`): the control run (C0,
  seed 0, pool 3090 Ti) reproduced round 3 exactly (same val curve to every digit, NaN at epoch 84). Its pre-LN sentence norms:
  median 1.5 -> 1e7, min -> 1e-20, max = inf (fp32 overflow) from epoch 22, on 20% of steps; those sentences give garbage embeddings
  and the gradient spikes (type gradient up to 165). At LR 3e-3 the same happens by epoch 6 (stress probe).
- Already at init a batch spans pre-LN norms 7e-4 .. 3e18; below ~0.07 the LayerNorm eps breaks scale invariance.

## 2. Candidate screen (phase 2; dec_t3, LR 1e-3 constant, 150 epochs, seeds 0 and 1; stress = 30 epochs at LR 3e-3, seed 0)
| candidate | stress NaN ep | NaN ep (s0 / s1) | overflow steps | best val mrr (s0 / s1) | test i2t R@10 (s0) | swap_full (s0) |
|---|---|---|---|---|---|---|
| C0 control | 6 | 84 / 65 | 20% / 17% | 0.016 / 0.014 | 0.045 | 0.866 |
| C1 grad clip 1.0 | 8 | - / - | 0.3% | 0.025 / 0.024 | 0.073 | 0.905 |
| C2 type LR x0.3 | 20 | - / - | 0% | 0.017 / 0.028 | 0.053 | 0.521 |
| C3a fixed norms, words+types | 4 | 27 / 29 | 8-11% | 0.011 / 0.012 | 0.039 | 0.548 |
| C3b fixed norms, types | 4 | 22 / 29 | 7-13% | 0.010 / 0.013 | 0.031 | 0.481 |
| C4 fixed norms + spectral cap 2 | 5 | 35 / 31 | 10-14% | 0.011 / 0.014 | 0.034 | 0.753 |
| C5 isometric type tensors | 22 | - / 141 | 11% | 0.030 / 0.032 | 0.097 | 0.589 |
| C6 MPS canonical gauge | 3 | 4 / - | 20% | 0.003 | 0.006 | 0.541 |
| C7 orthogonal init | 6 | 86 / 105 | 20% | **0.035** / 0.024 | **0.110** | 0.757 |
| C8 scale-free contraction | **-** | - / - | **0%** | 0.019 / 0.021 | 0.059 | 0.812 |
| C9 = C8 + fixed norms | | - / - | 0% | 0.016 / 0.044 | 0.050 | 0.843 |
| C10 = C8 + orthogonal init | (LR 3e-3: stable, no learning) | - / - | 0% | 0.125 / 0.034 (s2 0.033) | 0.318 | 0.943 |
| C11 = C10 + clip 1.0 | | - / - | 0% | 0.056 / 0.043 | 0.169 | 0.804 |
C6 was dropped from the long screen after its stress result (10x slower). Only the scale-free family never overflows.

## 3. Confirmation (phase 3) and the head fix (4b); test numbers from best.pt (val mrr), mean ± std over seeds
| | step-1 ref | step-3 dec_t3 (r2) | C0 (300 ep) | C10 (300) | C11 (300) | dec_t2 + C11 (300) | **C10 + plain head (150)** | C10 + no-bias head (150) | C11 + root bond 32 (150) | bag-of-words (150) |
|---|---|---|---|---|---|---|---|---|---|---|
| seeds / NaN | 3 / 0 | 3 / 0* | 3 / **2** | 3 / 0 | 3 / 0 | 3 / 0 | 3 / 0 | 2 / 0 | 2 / 0 | 1 / 0 |
| runs that jumped | - | 1 | 1 | 1 | 0 | 2 | **3** | **2** | 2 | - |
| coco5k i2t R@10 | 0.041 ± 0.004 | 0.124 ± 0.143 | 0.137 ± 0.160 | 0.209 ± 0.117 | 0.185 ± 0.040 | 0.268 ± 0.158 | **0.351 ± 0.007** | 0.349 ± 0.002 | 0.440 ± 0.002 | 0.759 |
| coco5k t2i R@10 | 0.009 ± 0.001 | 0.065 ± 0.094 | 0.074 ± 0.107 | 0.114 ± 0.107 | 0.077 ± 0.026 | 0.178 ± 0.128 | **0.246 ± 0.003** | 0.245 ± 0.011 | 0.305 ± 0.001 | 0.590 |
| SVO swap_full | 0.498 ± 0.011 | 0.851 ± 0.067 | 0.886 ± 0.030 | 0.774 ± 0.202 | 0.791 ± 0.079 | 0.891 ± 0.092 | **0.967 ± 0.018** | 0.966 ± 0.009 | 0.967 ± 0.005 | 0.205 |
| SVO-probes std | 0.509 ± 0.019 | 0.597 ± 0.101 | 0.602 ± 0.103 | 0.634 ± 0.079 | 0.625 ± 0.047 | 0.686 ± 0.093 | 0.727 ± 0.005 | 0.721 ± 0.002 | 0.756 ± 0.001 | 0.789 |
| ARO VG-Relation | 0.494 ± 0.007 | 0.535 ± 0.013 | 0.536 ± 0.011 | 0.517 ± 0.017 | 0.523 ± 0.004 | 0.543 ± 0.004 | 0.542 ± 0.004 | 0.547 ± 0.002 | 0.545 ± 0.005 | 0.256 |
| ARO VG-Attribution | 0.488 ± 0.014 | 0.527 ± 0.008 | 0.529 ± 0.013 | 0.505 ± 0.023 | 0.514 ± 0.020 | 0.537 ± 0.016 | 0.523 ± 0.010 | 0.527 ± 0.018 | 0.534 ± 0.001 | 0.250 |
| SC++ replace_object (chance 0.33) | 0.375 | 0.328 | 0.365 | 0.361 | 0.319 | 0.416 | 0.472 ± 0.004 | 0.483 | 0.463 | 0.858 |
| SC++ replace_relation | 0.341 | 0.293 | 0.270 | 0.298 | 0.297 | 0.289 | 0.295 | 0.297 | 0.328 | 0.516 |
| SC++ swap_attribute | 0.367 | 0.269 | 0.296 | 0.281 | 0.260 | 0.256 | 0.256 | 0.258 | 0.224 | 0.247 |
| Winoground text (chance 0.25) | 0.145 | 0.117 | 0.114 | 0.107 | 0.111 | 0.125 | 0.133 | 0.149 | 0.126 | 0.049 |
| text params | 538M | 18.0M | 18.0M | 18.0M | 18.0M | 18.0M | 18.0M | 18.0M | **139M** | 18.1M |
| peak train memory | 5.1 GB | 4.3 | 4.3 | 4.4 | 4.4 | 3.5 | 4.3 | 4.3 | **19.3** (OOM on 24 GB) | 3.2 |
\* step-3 round 2: runs cut by the plateau LR rule before they could diverge. 3-epoch budget: C10 and C11 are at chance like step 3
(coco i2t R@10 0.010 ± 0.001 / 0.010 ± 0.001; every compositional benchmark |z| < 1.3; `step4_tables.md`). The ± of the
bias-head columns is the jump lottery (one seed jumped, two did not); the head-fix columns have no such split.

## 4. Why some runs jump (B1-B5, head test)
| evidence | what it shows |
|---|---|
| Same seed, same config, pool 3090 Ti instead of the cluster GPU: no jump (val 0.053 vs 0.125) | the jump is not determined by the init; float-level differences decide it |
| Snapshots every 10 epochs (`s4_traj.py`): text effective rank 141 -> 4.7 by epoch 10, root output cores' stable rank 64 -> 2 by epoch 40, unrelated-caption cosine 0 -> 0.33, one image in 10k of 25k top-10 lists | the plateau is an early collapse of the embeddings |
| The post-head embeddings collapse first (epochs 10-30: pre-LN effective rank 70-82, after the head 9-10); head bias 0.13 -> 7.9 and gain 22.7 -> 8.3 by epoch 30 | the shared LayerNorm parameters move first |
| Final head bias norm: stuck runs 4.1-11.3, jumped 2.0-7.2, bag-of-words 1.35 | a tendency only (no clean separation); the intervention below decides |
| Root output cores forced isometric (rank 64 kept): still collapse, 0 of 3 jump | the root-core rank loss is a symptom, not the cause |
| Stuck run restarted at epoch 299 with new data order (x2) or 10% noise on every type tensor: 0 of 3 escape in 100 epochs | the collapsed state is a stable basin, not a saddle a nudge can tip |
| **Head without bias (gain kept) or without any parameters: 5 of 5 jump at epoch 50-80** | the shared bias causes it |
| Root bond 32 (bias kept): 2 of 2 jump, earlier and deeper (loss 2.9) | more output capacity lets the TN win against the shortcut |
Mechanism (consistent with all rows): the head's bias and gain receive a large, consistent gradient from every caption of every batch,
while each type tensor gets a diffuse one. The head therefore learns a shortcut first (a shared offset that rescales all image
scores) and drives the embeddings into a low-rank state; the TN then learns slowly inside it, and only some trajectories leave.

## 5. Why recall is low (A1-A4)
| | CLIP text encoder | bag-of-words | TN (C10 + plain head) | TN (C11 + root bond 32) | TN stuck (C10 s1) |
|---|---|---|---|---|---|
| coco5k i2t / t2i R@10 | 0.834 / 0.661 | 0.759 / 0.590 | 0.359 / 0.249 | 0.438 / 0.304 | 0.150 / 0.054 |
| effective rank of text embeddings (images: 65) | 68 | 51 | 22 | 35 | 7 |
| numerical rank per root type (pre-LN) | - | - | 64 (cap) | 470-500 | 64 |
| 1 - cos after swapping one word | - | 0.15 | 0.55 | 0.58 | 0.50 (32% of words ignored) |
| t2i R@10, captions <= 8 words / 13+ words | - | 0.57 / 0.65 | 0.27 / 0.21 | 0.32 / 0.26 | 0.06 / 0.04 |
| cos of same-image captions: same root / different root | - | 0.72 / 0.70 | 0.67 / 0.55 | 0.60 / 0.48 | 0.43 / 0.38 |
| max k-occurrence of an image in t2i top-10 (of 25k) | - | 408 | 783 | 453 | 11,716 |
Reading: the data and the frozen image space support ~0.76 R@10 with these word vectors. The TN loses most of that because (1) each
caption is written through an 8-dim bond on each side of its root word (<= 64 dims per root type; bond 32 removes the cap and gains
+25%); (2) its composition is multiplicative: every word is a factor, so one changed word moves the caption far, and longer
captions get worse instead of better; paraphrases that parse differently land in different root subspaces. (1) is a decomposition
question for step 5; (2) is the price of the order-sensitive composition that gives the TN its SVO/ARO advantage.

## 6. Efficiency (controlled: pool 3090 Ti, pinned, 2 epochs each, same seed; `tn_s4_prof_*`)
| | C0 control | C10 | C11 | C10 + plain head | C11 + root bond 32 |
|---|---|---|---|---|---|
| training samples/s (epoch 2) | 4,194 | 3,759 (-10%) | 3,779 | 3,769 | OOM (24 GB); L40S 19.3 GB peak |
| val pass | 2.0 s | 2.1 s | 2.1 s | 2.1 s | |
| peak memory | 4.33 GB | 4.33 | 4.33 | 4.33 | |
| text params | 18,040,040 | same | same | -1,024 (head) | 139,248,992 |
The -10% is the scale-free normalisation (one max-abs + divide per intermediate). Bag-of-words trains at ~52k samples/s.

## 7. Code and checks (all options default off; with defaults the losses are bit-identical to `tn-step3`: 444/444 steps)
- `text.scale_free`; `text.constraint` {norm, spectral_kappa, isometric, mps_gauge, init_orth, scope: root} (matmul-only, unit-tested
  against CPU SVD: `tools/tn/s4_constraints_test.py`); `grad_clip`; `type_lr_scale`, `type_weight_decay`, `decoupled_wd`;
  `text.head` (affine | no_bias | plain); `text.root_bond_dim`; `text.arch: bow`.
- `tn_stats` (trainstats/paramstats.csv, written before the divergence check), `save_state` + `scripts/train.py --resume`
  (optimiser rows restored exactly; caption sampling not bit-identical after a resume), `snapshot_state` + `--resume_as/--reseed/--perturb`.
- Tools: `tools/tn/s4_{norms,trainstats,metrics,embed,influence,paraphrase,traj,head,rootsr,clip_ceiling}.py`, lane runner
  `s4_lane.sh` (idempotent queues). Results: `results/tn/step4/{norms,recall,final,final_e3}`.

## Caveats
- Runs used mixed GPU types (pool 3090 Ti, cluster L40S/A100/3090/L4); metrics are comparable, only the controlled profile is used
  for speed. Whether a bias-head run jumps depends on the GPU's float behaviour (section 4), so those columns are noisy by nature.
- Bag-of-words and root bond 32 have 1 and 2 seeds; the head-fix runs are 150 epochs (they had converged: loss 3.2 at epoch 140).
- `C0 (300 ep)` mixes two NaN runs (reported at their last good epoch) and one survivor.

## Input for step 5 (decomposition)
- The root bond is a binding bottleneck (64 dims per root type, saturated in every good run); bond 32 on the root types alone gives
  +25% recall but 7.7x parameters and 19 GB. The decomposition study should vary the root/output bond separately from the rest.
- Run step 5 with the step-4 setup (scale-free, orthogonal init, no-bias head) so the decomposition isn't confounded by the jump lottery.

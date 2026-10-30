# TN retrieval, step 3: decouple word from complex type (2026-10-30)

Goal (user's spec): keep the existing tensor as the complex type only, add one wire to an order-1 word vector, compare
against the baseline where type tensors are not shared across words. Branch `tn-step3` (pushed to QLL). Full tables:
`step3_tables.md`. Raw outputs: tails `checkpoints/mscoco/tn/tn_s3*` (benchmark JSONs, run_meta, steps.csv, config),
`results/tn/step3/` (profiles, regression checks, summaries, plateau logs).

## Result
- **Decoupling works, generalises far better than the baseline, and has a large upside, but long training at full
  learning rate is numerically unstable.** The one decoupled run that trained at LR 1e-3 for 200 epochs without diverging
  (`dec_t3` seed 2, round 2, still improving) reaches coco5k i2t R@10 **0.289** and t2i R@10 **0.173** (baseline at its
  plateau 0.110 / 0.037), SVO-probes std **0.714**, SVO swap_full **0.925**, with **18M text parameters instead of 538M**
  and half the GPU memory. But 8 of 9 decoupled runs held at LR 1e-3 beyond ~65 epochs **diverged to NaN** (epochs 65-86;
  both thresholds; the same seed survives on one GPU and diverges on another), and runs whose LR was cut early by noisy
  validation are stable but stall at R@10 0.04-0.07. A real plateau for the decoupled model needs a stability fix:
  that is step 4.
- **Every decoupled run is far above chance on subject/object swaps** (SVO swap_full 0.59-0.93, z up to 41; ARO
  VG-Relation up to 0.55), where both baselines stay at chance (0.47-0.55) at every budget. No ties (ties count as wrong),
  real margins. The baselines never leave chance on any compositional benchmark.
- **Baselines plateau and overfit:** 538M parameters, train loss 3.2 at val mrr 0.035; `base` (glorot) and `base_norm`
  end within 0.005 R@10 of each other (0.110 / 0.105).
- Decoupled models are mostly below chance on SugarCREPE++ replace sets and on Winoground image/group (see "Caveats").
- Found and fixed on the way: a **scale bug** (glorot init puts sentence vectors at ~1e-6, decoupled ~1e-11, below the
  LayerNorm eps: decoupled text gradients were exactly 0), a **tie bug** and a **NaN bug** in the retrieval metrics.
  Training is **+34-36% faster** with bit-identical losses.

## Model
| | baseline (steps 1-2) | decoupled (step 3) |
|---|---|---|
| tensors | one per word@type (`dog_n`, MPS bond 8 for order >= 2) | one vector per word `w:dog` (dim 128, all uses), one tensor per complex type `t:n.r@s@n.l` (all words of that type) with one extra leg of dim 128 in the middle of its legs |
| decomposition | every word@type tensor of order >= 2 | type tensors only: `dec_t2` order >= 2 (word leg counted), `dec_t3` order >= 3 |
| text parameters | 537,860,382 (133,903 tensors) | dt2 17,997,936 / dt3 18,040,040 (~29k word vectors + ~1.9k type tensors) |
| peak GPU memory, training | 8.0-9.2 GB | 3.4 / 4.3 GB |
| OOV test sentences (coco5k) | 18.8% | 9.7% (an unseen word now only needs a vector; its type tensor is trained) |

Code: `TNCompiler(decouple=True)` (`text.decouple`, `text.word_dim`); configs `configs/tn/{dec_t2,dec_t3,ref_norm}.yaml`.
Checks: `tools/tn/check_decouple.py` (contraction = hand-written einsum; a word used with two types shares one vector);
`tools/tn/regress_compile.py` (decouple off: einsums, symbols and compiler id identical to step 2 on 20k rows).

## Scale bug and the norm-preserving init (user decision: apply to all)
Glorot init makes each contraction shrink the norm (~17x per 3-leg word in an MPS chain). Sentence vectors before the
LayerNorm head: baseline median ~8e-6, decoupled ~4e-12 (some underflow to exactly 0), far below LN eps 1e-5, so LN outputs
~its bias: decoupled text gradients were exactly 0 and the loss flat; the baseline was already in that regime (likely
why steps 1-2 "barely trained in 3 epochs"). Fix `text.init: norm`: std 1/sqrt(dim of the tensor's leg towards the
sentence output) (the compiler marks that leg per symbol by a walk from the output wire), so each tensor maps unit-norm
inputs to a unit-norm output in expectation: median sentence norm 1.1-2.0 at init, text grads 0.9-1.5. `base_norm` is the
step-2 model with this init (default stays glorot).

## Speed (per change, pool RTX 3090 Ti pinned to P-cores, 1 epoch = 222 steps, seed 0)
Every change: 222-step loss trace bit-identical to the pre-step-3 code (`tools/tn/regress.py`, `s3_chain.sh`); the
planner change also passes `tools/tn/check_schedule.py` (identical index arrays and specs on 450 batches).

| change | base_norm samples/s | dec_t3 samples/s | text fwd ms/step (base / dt3) | val pass s (base / dt3) |
|---|---|---|---|---|
| before (9caeb47) | 4,893 | 3,050 | 56.6 / 96.8 | 5.8 / 7.5 |
| CocoDataset plain lists, not a pandas row per sample (a987435) | 5,319 | 3,224 | 57.2 / 97.3 (data 11.9 -> 2.7 ms) | 5.5 / 7.3 |
| eval on the GPU, ranks by counting, no 5k x 25k argsort (36c71b0) | 5,290 | 3,258 | 56.2 / 96.2 | 2.5 / 4.3 |
| vectorised levelwise planner, row refs as ranges (52e39ed) | 6,105 | 4,217 | 43.8 / 61.7 | 1.9 / 2.9 |
| `gc.freeze()` of the compiled captions after text_init (8de13be) | **6,534 (+34%)** | **4,134 (+36%)** | 39.9 / 63.3 | 1.7 / 2.9 |
| parallel caption compile (874f021) | no gain; startup 33 -> 50 s | | | reverted |

The planner gain only showed for dt3 together with `gc.freeze`: without it, full cyclic-GC passes over the millions of
long-lived caption objects ate it (dt3 3,197 vs 4,134 samples/s). Peak memory unchanged (6.8 / 1.4 GB in the profile).
Still dominant for dt3: backward 52 ms/step (baseline 31) and forward kernel launches.
On the cluster the same code is ~2.2x slower (CPU-bound): use pool GPUs for TN work when they are free.

## Metrics fixes
- Ties: duplicate captions give identical text vectors, so exact score ties are common (2.5-4% of coco5k images). Rank
  counting alone counted ties for the positive (i2t R@1 0.0062 -> 0.0112 on one checkpoint). Now every retrieval metric
  is its expectation under random tie-breaking (R@K = P(rank < K), MRR = E[1/rank]): equal to the old argsort metric
  without ties, its average over tie orders with them (`tools/tn/check_recall.py`); old vs new on fixed checkpoints
  within 6e-4. `*_tie_frac` is reported.
- NaN: a diverged model's NaN scores compared false everywhere and scored "perfect" (dt2 s1 val mrr 1.0, best.pt
  overwritten by the NaN model). NaN scores now rank at chance, and training stops on a non-finite epoch loss before
  saving (best.pt keeps the last good model).

## Results
Selection on validation only (val mrr_mean = mean of i2t/t2i MRR on the 5k-image holdout); test numbers from best.pt.
Mean ± std over 3 seeds; per-seed values and every benchmark in `step3_tables.md`.

### 3 epochs (cosine), all on L40S except the step-2 reference
| | step-2 ref | base | base_norm | dec_t2 | dec_t3 |
|---|---|---|---|---|---|
| coco5k i2t R@10 | 0.041 ± 0.004 | 0.041 ± 0.004 | 0.032 ± 0.001 | 0.009 ± 0.001 | 0.010 ± 0.002 |
| coco5k t2i R@10 | 0.009 ± 0.001 | 0.009 ± 0.001 | 0.007 ± 0.001 | 0.003 ± 0.000 | 0.003 ± 0.000 |
| compositional (ARO, SVO, SugarCREPE++) | chance | chance | chance | chance | chance |
`base` on the L40S reproduces the step-2 3090 Ti reference. 3 epochs is far too short for any model here.

### Until plateau
Round 1: LR x0.1 after 4 epochs without val gain, stop after 10. Round 2: after 10 / 20, max 200 (all configs).

| | base R1 | base R2 | base_norm R1 | base_norm R2 | dec_t2 R1 | dec_t2 R2 | dec_t3 R1 | dec_t3 R2 |
|---|---|---|---|---|---|---|---|---|
| epochs trained | 59-64 | 68-126 | 45-56 | 70-128 | 45-71 | 71-85 (all NaN) | 36-76 | 108-200 |
| coco5k i2t R@10 | 0.108 ± 0.006 | 0.110 ± 0.008 | 0.098 ± 0.003 | 0.105 ± 0.004 | 0.052 ± 0.024 | 0.066 ± 0.012 | 0.048 ± 0.023 | 0.124 ± 0.143 |
| coco5k t2i R@10 | 0.034 ± 0.003 | 0.037 ± 0.004 | 0.030 ± 0.001 | 0.034 ± 0.002 | 0.014 ± 0.008 | 0.021 ± 0.002 | 0.013 ± 0.009 | 0.065 ± 0.094 |
| ARO VG-Relation | 0.499 | 0.494 | 0.503 | 0.501 | 0.527 (z 3.3) | 0.524 (z 2.9) | 0.533 (z 4.0) | 0.535 (z 4.2) |
| ARO VG-Attribution | 0.494 | 0.508 | 0.496 | 0.500 | 0.514 | 0.514 | 0.505 | 0.527 (z 3.5) |
| SVO-probes std (image choice) | 0.503 | 0.524 | 0.513 | 0.506 | 0.530 | 0.550 (z 4.3) | 0.520 | 0.596 (z 8.4) |
| SVO swap_full (text choice, 1777) | 0.491 | 0.525 | 0.503 | 0.532 | 0.711 (z 19.6) | 0.724 (z 21.2) | 0.802 (z 32.0) | 0.851 (z 41.5) |
| SugarCREPE++ (5 sets, chance 1/3) | 0.32-0.35 | 0.32-0.37 | 0.33-0.34 | 0.33-0.35 | 0.28-0.31 | 0.28-0.31 | 0.28-0.30 | 0.27-0.33 |
| Winoground group (chance 1/6) | 0.050 | 0.068 | 0.082 | 0.062 | 0.041 | 0.020 | 0.035 | 0.028 |

dec_t3 round 2 per seed: i2t R@10 0.043 / 0.041 / **0.289**, t2i R@10 0.009 / 0.013 / **0.173**, SVO std 0.536 / 0.540 /
**0.714**, swap_full 0.831 / 0.796 / **0.925**. Validation curves: seeds 0 and 1 had their LR cut to 1e-4 at epochs ~20-40
(val mrr_mean 0.008-0.010 at the time) and then crawled to 0.013; seed 2 kept 1e-3 for all 200 epochs and rose steadily
(val mrr_mean 0.012 at epoch 20, 0.038 at 100, 0.077 at 140, 0.107 at 199). base_norm plateaus at val mrr ~0.035 with train
loss 3.2 (538M parameters, overfits); dec_t3 seed 2 reaches 0.107 at train loss 4.0.

### Round 3: dec_t3, LR x0.1 only after 30 epochs without val gain, stop after 40, max 500 (pool 3090 Ti)
All three seeds diverged to NaN at full LR (epochs 84, 65, 86); reported = best.pt before divergence (epochs 75, 63, 82).
| | seed 0 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| coco5k i2t R@10 | 0.051 | 0.045 | 0.102 | 0.066 ± 0.031 |
| coco5k t2i R@10 | 0.014 | 0.012 | 0.037 | 0.021 ± 0.014 |
| SVO swap_full | 0.872 | 0.872 | 0.588 | 0.777 ± 0.164 (z 28) |
| SVO-probes std | 0.538 | 0.541 | 0.544 | 0.541 (z 3.6) |
| ARO VG-Relation | 0.535 | 0.525 | 0.496 | 0.518 |
Seed 2 followed its round-2 trajectory (val mrr_mean 0.034 at epoch 85 vs 0.030 at epoch 80 in round 2) until it diverged.

### Divergence
| runs held at LR 1e-3 past ~65 epochs | diverged |
|---|---|
| dec_t2: round 1 seed 1 (+ its rerun s1b), round 2 seeds 0-2 | 5 of 5 (epochs 71, 71, 71, 74, 85; seed 1 always at 71) |
| dec_t3: round 2 seed 2, round 3 seeds 0-2 | 3 of 4 (epochs 65, 84, 86; round 2 seed 2 survived 200 epochs on a cluster GPU) |
| base, base_norm (all rounds) | 0 (their LR is cut by epoch ~50-100: they plateau) |
| (not counted: runs whose LR was cut before epoch ~65, e.g. round-1 dec_t3 rerun s2b, cut after epoch 45) | |
Gradient-norm spikes (>5, up to 27) precede it. The decoupled model shares one tensor per complex type across every
word, so a type tensor gets gradient from a large share of every batch and its scale can run away; the dense-noun
threshold (dec_t3) delays but does not prevent it.

## Caveats
- The best decoupled result rests on one run that happened not to diverge; it shows what the model can reach, not
  what it reliably reaches today.
- The plateau rule interacts with how noisy and slow the validation metric is: the decoupled models need a more
  patient rule than the baselines. Same rule for all configs within a round, so each round is a fair comparison.
- SugarCREPE++ replace sets and Winoground: decoupled models are below chance. Both benchmarks pair two positive
  captions (or two images) with hard negatives that reuse the same words; a model that scores word content strongly
  (shared word vectors) but composes weakly can systematically prefer the negative. Not investigated yet.
- Mixed GPU types across runs (pool 3090 Ti, cluster L40S / RTX 6000 Ada / 3090): metrics are unaffected beyond float
  drift; speed numbers come only from the controlled per-change profiles above.
- Diverged runs report their last good checkpoint before NaN.

## Input for step 4 (regularisation and constraints)
- Divergence of the shared type tensors at LR 1e-3 (norm bounds / spectral constraints on type tensors, gradient
  clipping, per-group LR for shared vs per-word parameters); the norm-preserving init is a first data point for init.
- The plateau LR rule fires on validation noise for the slow decoupled models: step 4 should compare candidates at a
  fixed long budget (or with a smoothed selection metric) so the schedule does not decide the comparison.

## Compute and housekeeping
3-epoch and plateau runs on the CS cluster (one job per run) and pool 3090 Tis. Pool node list in
`comp_bridge/scripts/_common.sh` was missing 8 nodes that `check_nodes.sh` knows (brent, mandarin, ...): synced. brent and
gadwall GPUs fail CUDA init. The pool switched to Windows at ~19:45 and killed one run (rerun). Cluster checkpoints were
copied to tails, md5-verified and deleted on the cluster (user-approved).

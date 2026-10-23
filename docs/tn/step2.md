# TN retrieval, step 2: efficiency and speed (2026-10-23)

Goal: make TN training and inference as fast as possible before the model changes (steps 3-6), without changing
the model. Every change was regression-checked against the step-1 reference. Tables: `step2_tables.md`
(per-change speed, 3-seed metrics). Raw outputs: tails `results/tn/step2/` (`prof_<tag>.json`, `bench_<tag>.json`,
`regress*_<tag>.txt`, `summary/`). Branch `tn-step2`.

## Result
| | step 1 (baseline code) | step 2 (`text.tn_engine: levelwise`) | speed-up |
|---|---|---|---|
| training, pinned 3090 Ti, epoch 0 (222 steps) | 241 samples/s, 2.13 s/step | 4,839 samples/s, 0.106 s/step | **20x** |
| one epoch of full MSCOCO (3-epoch runs, run_meta) | 508 s | 24 s | 21x |
| val eval per epoch (25k captions) | 29-31 s | 4.8 s | 6x |
| benchmark of one checkpoint, all 12 test sets | 141 s (eval alone 56 s) | 55-57 s (eval alone 12 s) | 2.5x (4.8x) |
| text parameters | 537,860,382 | 537,860,382 | same |
| peak GPU memory, training | 5.1 GB | 8.0 GB | +2.9 GB |

A reference run (3 epochs + benchmark) now takes ~5 min per seed on one pool GPU (3 seeds: 15.8 min), vs ~30 min.
Same model and metrics: 3-seed step-2 means match step 1 to within ±0.003, far inside the seed spread
(table below).

## Setup
- Compute target: pool RTX 3090 Ti (shoveler, mallard; same CPU i9-12900K). Training is CPU-bound (Python and kernel
  launches), and the 12900K has slow E-cores: unpinned runs of identical code varied 225-242 samples/s. All
  measurements are pinned to the P-cores (`taskset -c 0-15`). Repeat runs of unchanged training code agree within 1%
  (241/242, 4809/4839), with one outlier (change 11: 3891, rerun as change 12).
- Baseline: commit `addfbe3` (step-1 code plus harness hooks that don't change behaviour). Step-1 3090 Ti reference
  run `tn_s1_clip_s0` (code 86eb697 = ccd1411 training code).
- Per-change measurement (`tools/tn/check_change.sh`, run in a per-commit worktree by `tools/tn/wt_check.sh`):
  `tools/tn/profile_train.py` (train.py setup, seed 0, all 222 steps of epoch 0, cuda-synchronised phase times,
  cProfile + torch.profiler hotspots) and `scripts/benchmark.py` on `tn_s1_clip_s0/last.pt`.
- Regression (`tools/tn/regress.py`): the 222-step loss trace vs the baseline's (tolerance 1e-5); the benchmark per-item
  scores (23,506 ARO/SVO scores, tolerance 1e-5) and metrics (tolerance 1e-3: accuracies and recalls are counts, so a
  rounding-level score change can flip a near-tie).

## What the baseline spent its time on (2.24 s/step)
~9,400 small pairwise einsums per step: 415 sentence structures per batch × ~22 pairwise steps each (opt_einsum
path), so ~28k bmm launches at ~33 µs of CPU each, forward and backward. The GPU was busy ~0.2 s per step. Then
`torch.stack` of ParameterDict tensors (~10k/step), Adam over ~100k separate parameters (~100 ms) and index_put
output assembly.

## Changes, each measured separately (pinned 3090 Ti; `step2_tables.md` has the full table)
| # | change (commit) | samples/s | vs previous | regression |
|---|---|---|---|---|
| - | baseline (addfbe3 / pinned 0889364) | 241 | - | - |
| 1 | benchmark reads the vocabulary from the checkpoint, no train/val recompile (3ccfabb) | 242 (training unchanged) | benchmark startup 58 → 9 s | exact |
| 2+4 | per-sentence plans cached; one gather puts outputs back in batch order (4326aa2) | 247 | 1.02x | exact |
| 3 | packed parameter banks (one per shape) + RowAdam (c482aeb) | 311 | 1.26x | losses ≤ 4.8e-7, metrics exact |
| 6 | levelwise engine: each structure's pairwise steps run for all sentences at once (e0fcf45) | 566 | 1.82x | losses ≤ 9.5e-7 |
| 7 | one gather per (round, shape) with a copy-only backward (a2cf252) | 984 | 1.74x | losses ≤ 9.5e-7 |
| 8 | contraction paths searched up front in a fork pool; no pin_memory for TN (212227b) | 1,179 | 1.20x | losses ≤ 9.5e-7 |
| 9 | rounds by dependency depth, equations keyed by axis positions (44f7f4f) | 3,543 | 3.01x | losses ≤ 9.5e-7 |
| 10 | level plans built in the pool too (0e9b66c) | 4,809 | 1.36x | losses ≤ 9.5e-7 |
| 11 | test-set plans up front in benchmark; vectorised retrieval mask (349d7d8) | (3,891: outlier) | eval 18.7 → 11.7 s | as 10 |
| 12 | in-process search below 4,000 structures (eb67fc7) | 4,839 | - | as 10 |

Changes 6-12 sit behind `text.tn_engine: levelwise`; the default (`grouped`) is the step-1 engine plus changes 1-3.
"Regression" for 6-12: 174/222 losses bit-identical, the rest ≤ 9.5e-7; ARO/SVO scores ≤ 6.7e-8; 84/88 metrics
identical, and the other 4 are coco5k i2t, where 2 of 5,000 images change their rank-1 caption (0.0004).

How the levelwise engine works: a 512-caption batch has ~415 structures, but a structure is a tree of pairwise
contractions. Each structure's opt_einsum path (the same one the grouped engine uses) is replayed level by level:
round t runs every pairwise step at dependency depth t+1 across all sentences, and steps that contract the same axis
positions of the same shapes run as one batched einsum. A batch becomes ~10 rounds and ~230 einsums, down from ~9,400.
Word tensors are expanded once per step (a row per use), so each intermediate row is read exactly once and every
gather has a copy-only backward. The schedule is planned on the CPU and its indices go to the GPU in one copy. The
maths per sentence is unchanged; only the batching of the bmm kernels differs, hence rounding-level differences.

RowAdam (change 3): Adam over one bank per tensor shape, updating only the rows a step used, with a step count per row.
This is exactly `torch.optim.Adam` with one tensor per word (untouched tensors are skipped and keep their own
bias-correction count). In a CPU float64 toy test it matches the old model to 7e-15. Checkpoints keep the
`params.<symbol>` keys: old checkpoints load unchanged, and new ones load in old code.

Tried and dropped: an on-disk cache of the compiled datasets (3ccfabb, removed in fad08fb). The compiled train set
pickles to 808 MB and loads in ~47 s, slower than recompiling (29 s).

Memory: +2.9 GB peak in training, because Adam's moments now exist for whole banks from the first step (2 × 2.2 GB;
before, they grew as tensors were first touched), plus the per-step pools. 8.0 GB of 24 GB.

## Full-run regression (3 epochs, train.py, seed 0, same GPU type)
`tn_s2_opt_s0` (levelwise, f2ea30f) vs `tn_s1_clip_s0` (step 1, 3090 Ti): losses agree to 1e-6 for the first 368 steps,
then rounding differences grow (> 1e-5 from step 410, max 2.2e-3 at step 664). That is the same drift as **identical
step-1 code on two GPU types** (`tn_s1_clip_s0` vs `tn_s1_clip_h_s0`: > 1e-5 from step 411, max 2.2e-3). Benchmark
metrics: max difference 0.014 (step 2 vs step 1 on 3090 Ti) and 0.008 (vs step 1 on L40S); identical code across GPUs
gives 0.010. So step 2 is as close to the reference as the reference is to itself on different hardware.

## Step-1 vs step-2 metrics (seeds 0-2; step 1 = HPC runs `tn_s1_clip_h_s*`, step 2 = `tn_s2_opt_s*` on 3090 Ti)
| benchmark | chance | step 1 | step 2 |
|---|---|---|---|
| MSCOCO 5k i2t R@1 / R@5 / R@10 | 0.0002 / 0.0010 / 0.0020 | 0.0065±0.0008 / 0.0251±0.0039 / 0.0414±0.0040 | 0.0071±0.0011 / 0.0250±0.0043 / 0.0407±0.0039 |
| MSCOCO 5k t2i R@1 / R@5 / R@10 | 0.0002 / 0.0010 / 0.0020 | 0.0011±0.0002 / 0.0047±0.0006 / 0.0088±0.0008 | 0.0011±0.0002 / 0.0047±0.0006 / 0.0088±0.0008 |
| ARO VG-Relation / VG-Attribution | 0.5 | 0.4944±0.0066 / 0.4879±0.0143 | 0.4946±0.0064 / 0.4879±0.0143 |
| SVO std / swap_full / swap_animate | 0.5 | 0.5090±0.0185 / 0.4975±0.0105 / 0.5697±0.0378 | 0.5090±0.0185 / 0.4971±0.0104 / 0.5697±0.0378 |
| SugarCREPE++ swap_obj / swap_att / repl_obj / repl_att / repl_rel | 0.333 | 0.333 / 0.367 / 0.375 / 0.363 / 0.341 | 0.336 / 0.366 / 0.377 / 0.361 / 0.341 |
| Winoground text / image / group | 0.25 / 0.25 / 0.167 | 0.145±0.015 / 0.151±0.014 / 0.064±0.000 | 0.148±0.014 / 0.151±0.014 / 0.065±0.002 |
SugarCREPE++ stds (step 1 / step 2): 0.043/0.038, 0.015/0.017, 0.023/0.023, 0.007/0.004, 0.011/0.011. The conclusions
of step 1 stand: retrieval above chance, compositional benchmarks at chance, and 3 epochs barely train the model
(final train loss ~ln(512)).

## Vocabulary normalisation (your addition)
Already in place, so nothing changed: `modules/symbolic/grammar.py:strip_sent` lowercases and strips punctuation before
Bobcat parses. A scan of every parsed set (MSCOCO train/val_holdout/validation, SugarCREPE, Winoground, ARO, SVO) found
0 uppercase and 0 punctuation tokens (train: 28,386 words, 55,440 word@type symbols), so 'Tree', 'tree.' and 'tree' are
one symbol. Decided not to lemmatise (it would merge 'dogs'/'dog', 'is'/'are'): morphology is for step 6.

## Remaining bottlenecks
- Training step (106 ms): text forward 57 ms (mostly CPU planning of the schedule), backward 31 ms, data 12 ms
  (pandas row access in the dataset), optimizer 4 ms. The GPU is now the smaller part.
- Start-up (~50 s per run): compiling MSCOCO (~28 s, pure-Python TNCompiler), plan search (~9 s, parallel), init.
  Benchmark: compiling the 12 test sets (~25-35 s) is now its largest part.

## Use
- New runs: add `text: {tn_engine: levelwise}` (or `--set text.tn_engine=levelwise`). `scripts/benchmark.py --set ...`
  overrides eval config values. `tools/tn/seq2.sh <tag> <seeds> -- --set ...` = train + benchmark, pinned.
- Speed checks: `bash tools/tn/wt_check.sh <tag> <commit> [steps] [--set ...]`; tables: `tools/tn/step2_table.py`,
  `tools/tn/summarise.py`.

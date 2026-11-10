# Step 5: decomposition of the type tensors (DRAFT, in progress)

Branch `tn-step5` (off `tn-step4c`), worktree `comp_bridge/qclip_wt/step5`. Plan: `~/.claude/plans/tn-step5-decomposition.md`.
Base: o16v3 from step 4c (shared output core, output bond 16, centred base-point words), step-4 training setup, 150 epochs,
best.pt by val mrr. rsum = R@1 + R@5 + R@10 over i2t and t2i, in % (max 600).

## 1. Speed of the centred pass (sub-step 1)
`text.centre` now runs the real and the base (structure) rows in one levelwise schedule (6db9eb9). One RTX 3090, 300 steps, same seed:
| | s/step old | s/step new | losses |
|---|---|---|---|
| o16v3 | 0.871 | 0.601 (-31%) | max rel. diff 1.5e-7 (not bit-identical: bigger bmm buckets, one gradient leaf per bank) |
| o16 (path unchanged) | 0.422 | 0.488 | 222/222 bit-identical (the 15% is cluster CPU noise) |
tools/tn/s5_merge_test.py: outputs and row gradients equal the two-pass code to 3e-6 relative. Training is launch-bound
(per 10 steps: CPU 10.4 s, CUDA 1.6 s old; 5.5 s / 1.3 s new).

## 2. Core sharing across complex types (S screen, MPS)
`text.share_args` (bd8015a): every argument core (all but the word-leg core and the output core) is one tensor shared by all
complex types, keyed by (atomic leg type, side of the word leg, core shape) plus the distance from the word leg (S1, `leg_dist`)
or not (S2, `leg`). The word-leg core stays per type. In the o16v3 vocabulary: ~1.5k argument cores (9.5M params) -> 75 (S1).
2 seeds each; S0 = o16v3 (step 4c).
| | step 4 | S0 (o16v3) | S1, bond 8 | S2, bond 8 | **S1, bond 16** | bag-of-words |
|---|---|---|---|---|---|---|
| i2t / t2i R@10 | 0.350 / 0.245 | 0.415 / 0.304 | 0.413 / 0.305 | 0.405 / 0.299 | **0.541 / 0.409** | 0.759 / 0.590 |
| i2t R@1 | 0.071 | 0.095 | 0.092 | 0.091 | **0.152** | 0.367 |
| rsum | 109.2 ± 2.1 | 135.3 ± 0.5 | 135.0 ± 0.7 | 131.9 ± 0.6 | **187.6 ± 1.2** | 305.3 |
| SVO swap_full | 0.966 | 0.976 | 0.957 | 0.937 ± 0.040 | 0.963 | 0.205 |
| ARO-Rel / ARO-Att | 0.544 / 0.525 | 0.548 / 0.525 | 0.546 / 0.523 | 0.548 / 0.514 | 0.535 / 0.534 | 0.256 / 0.250 |
| SVO std | 0.725 | 0.748 | 0.752 | 0.732 | 0.761 | 0.789 |
| SC++ replace-object | 0.476 | 0.532 | 0.527 | 0.531 | 0.608 | 0.858 |
| text params | 18.0M | 16.2M | 7.5M | 7.0M | 14.1M | 18.1M |
| peak train memory (GB) | 4.3 | 5.5 | 8.8 | 8.8 | 13.4 | 3.2 |
Reading: sharing the argument cores costs nothing at bond 8 (S1 = S0 at 46% of the parameters); dropping the distance (S2)
costs 3 rsum and makes word order less reliable. With the parameters freed, bond 16 (all internal bonds, still fewer parameters
than o16v3) gives +39% rsum and +30% / +35% R@10 while keeping word order (swap_full 0.963) and ARO; ARO-Rel dips 0.013.
Val mrr is flat from epoch ~45 (0.24, no decline); train loss 2.48 (bond 8: 2.89). Peak memory rose with sharing (5.5 -> 8.8 GB
at equal bond; mixed GPUs, includes the validation pass): not yet explained, to check in the controlled profile.
Bond 16 OOM'd a 24 GB GPU in validation (one no-grad forward of ~2.5k captions); `text.eval_chunk` (eeb5ffd) fixes it.
Tables: results/tn/step5/screenS/summary.md.

## Next
Internal-rank sweep S1 bond 24 / 32 (running), decomposition families (CP, Tucker, tensor ring) at matched parameters on S1.

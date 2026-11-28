# Step 7A: compositional image graph from PSG ground-truth scene graphs

Branch tn-step7 (code d136ec7 for every run; worktree tails `qclip_wt/step7`). Plan: `~/.claude/plans/shiny-doodling-aurora.md`.

## 1. Summary
- **A scene-graph tensor network on the image side matches, but does not beat, the frozen CLIP image vector on retrieval.**
  On the 2,170 PSG ∩ coco5k test images (rsum, 6 recalls): R0 (frozen CLIP) 340.6; G1 (full graph) 332.4; G2 (objects + apex,
  no relations) 333.8. G1 ≈ G2: relations add nothing to retrieval. Seed spread is large (G2: 341.0 / 326.7), so G1 vs G2 vs R0
  are within noise except that no graph model is above R0.
- **The graph alone carries most of the signal:** G3 (no global CLIP vector; objects = crop + class, relations, apex base only)
  reaches 282.2, 83% of R0, from object crops and classes composed by the tensor network.
- **The model reads objects, not relations (per-error cost, the headline).** One injected error per test image costs: wrong
  object class 0.06 R@10, a missing object 0.03, a spurious or relabelled relation 0.02-0.03, a dropped or reversed relation
  ~0. Reversing a relation (subject ↔ object) is free: direction is ignored.
- **Relations are learned but not used.** The predicate matrices R_p moved far from their identity init (|A−I|/|I| 1.1),
  differ across predicates (relative spread 0.69) and are 96% off-diagonal, yet removing one costs nothing. Retrieval on COCO
  does not reward relational structure: the image-side counterpart of the bag-of-words finding for contrastive VLMs
  (Yuksekgonul et al., ICLR 2023, arXiv 2210.01936). In thesis terms (requirement I): **the retrieval training signal does not
  identify the shared relation generators**; they are free directions of the loss.
- **First non-tree graph family in qclip, contracted exactly.** Scene graphs have treewidth 0-4 (31% ≥ 2; +1 with the apex).
  The cost is set by treewidth as M2 predicts (uncapped, D = 32, A = 128): flops per image p50 2·10⁶; the 31% of graphs with
  treewidth ≥ 3 including the apex carry 93% of all flops, the 1.3% with ≥ 4 carry 63%, and those 612 graphs needed a cap (below).

## 2. Model (modules/models/vision/graph_tn.py)
Per image: object j has leaf o_j = base + P·crop_j + E[class_j] ∈ R^D (crop = frozen CLIP B/32 of its box, squared); every
object is a copy tensor (one index k_j). A relation (s, p, o) is R_p = A·diag(m_p)·B on (k_s, k_o) (A, B shared; identity
init). An apex node with leaf g = base_g + P_g·CLIP_global holds the open index z and joins every object through a shared
S (D × A_out). Z(G)[z] = g[z] Σ_k Π_j o_j[k_j] S[k_j, z] Π_rel R_p[k_s, k_o], then a linear head to 512 and unit norm.
D = 32, A_out = 128; 160k image parameters. Text side: the step-5 recipe (CP128 + R1, word_base 0.3 + centre, learnable
temperature + v1 negatives, EMA 0.999, 150 epochs, best by val MRR), trained jointly from scratch on the PSG subset.
- Contracted by the text side's levelwise engine: object / global leaves are data leaves ('x:'/'y:' banks with values from
  `data_fn`), other tensors are banks with dense gradients. Engine output equals a per-graph opt_einsum reference (rel. error
  1e-5, cycles included); relations-off output equals the closed form g ⊙ Π_j(Sᵀo_j); text-only training is bit-identical to
  tn-neutral (`tools/tn/s7_graph_check.py`, cluster smoke 7857905).
- **Contraction paths:** opt_einsum's default search gave paths up to 1e31 flops on 80-object graphs of treewidth 3. Explicit
  paths from a min-degree elimination order (`elim_path`) bound every graph: max 6.6·10¹⁰ flops.
- **Treewidth cap (`vision.max_tw: 2`):** at D = 32, A = 128 the tw ≥ 3 graphs have intermediates up to 4·10⁹ entries (OOM).
  Relations are dropped greedily until tw ≤ 2: 612 graphs (1.3%), 1,318 of 275,054 relations (0.5%). (Plan said tw ≤ 3; that
  still OOMs at D = 32.) 0.15 s per 512-image batch forward + backward, 3.5 GB (7.8 GB on the costliest batch).

## 3. Data (tools/tn/s7_psg_graphs.py, s7_crops.py; data/psg/)
PSG (psg.json, COCO ∩ VG, 133 classes, 56 predicates): 48,749 graphs, 538,153 objects, 275,054 relations (317 self-loops
dropped). Treewidth (min-degree bound) 0 / 1 / 2 / 3 / 4: 876 / 32,850 / 14,411 / 598 / 14 graphs. Splits = our caption tables
restricted to images with a graph: train 44,386 (of 113,287), val 1,952, test 2,170 (of coco5k); no test image in train/val.

## 4. Results (rsum on the 2,170-image PSG test pool; compare only within this pool)
| run | image side | rsum | seeds | i2t R@1/5/10 | t2i R@1/5/10 | val-best epoch |
|---|---|---|---|---|---|---|
| R0 | frozen CLIP vector | **340.6** | 339.0, 342.2 | 35.4 / 69.5 / 81.5 | 25.6 / 57.3 / 71.3 | 97, 78 |
| G1 | graph (objects + relations + apex) | 332.4 | 335.6, 329.2 | 32.3 / 66.0 / 79.2 | 25.2 / 57.5 / 72.2 | 137, 130 |
| G2 | G1 without relations | 333.8 | 341.0, 326.7 | 32.5 / 65.6 / 79.8 | 25.8 / 57.8 / 72.4 | 142, 137 |
| G3 | G1 without the global vector | 282.2 | 1 seed | 23.5 / 55.5 / 69.3 | 19.4 / 49.2 / 65.3 | 144 |

The graph models are slightly better on t2i (R@10 72.2-72.4 vs 71.3) and worse on i2t (79-80 vs 81.5). They select their best
epoch at 130-144 of 150 (R0 at 78-97): still improving, so 150 epochs may undertrain the image side.

**Per-error cost** (`tools/tn/s7_corrupt.py`; mean R@10 drop on the images an error applies to; G1 = mean of 2 seeds):

| error (one per image) | G1 t2i | G1 i2t | G2 t2i | G2 i2t |
|---|---|---|---|---|
| drop one relation | −0.002 | +0.000 | 0 | 0 |
| relabel one predicate | +0.022 | +0.019 | 0 | 0 |
| reverse subject/object | +0.002 | −0.001 | 0 | 0 |
| add a spurious relation | +0.030 | +0.026 | 0 | 0 |
| drop one object | +0.031 | +0.023 | +0.032 | +0.033 |
| wrong object class (crop kept) | +0.059 | +0.057 | +0.067 | +0.062 |

(G2 has no relation tensors: relation errors are no-ops, a check of the tool.) Relations act as noise: a wrong or extra one
costs, a missing one does not. **Learned relations** (`tools/tn/s7_relstats.py`, G1 s0): |A−I|/|I| 1.12, |B−I|/|I| 1.04,
m_p mean 0.13 (std 0.23); predicates differ (mean |R_p − mean R| / |mean R| 0.69); 96% of each R_p is off-diagonal.

Compute: R0 6.0 h (L4), G1 4.3 h (RTX 3090), G2 ~2.2 h (pool 3090 Ti), G3 4.2 h; GPUs differ, so wall times do not compare.

## 5. What it means
- **For the step-7 aim (compositional image side):** with ground-truth graphs the TN image side reaches the frozen CLIP vector
  but its relational part is unused, because nothing in COCO retrieval needs it. Retrieval is the wrong test for relations;
  the relation-sensitive benchmarks (ARO-Rel, SVO swaps, SugarCrepe replace-relation, Winoground) need graphs for their images:
  phase B (REACT++ generator).
- **For the thesis table:** first image-side graph family beyond trees (scene graphs + apex, tw ≤ 3 after the cap), generic
  generators, linear readout. Exact classical contraction works because PSG graphs have tiny treewidth; the cost concentrates in
  the high-treewidth tail exactly as M2's D^(tw+1) bound says, and the tail already forced a cap at D = 32. Identifiability (I)
  fails for the relation generators under a retrieval loss: the loss is flat along them.
- **Most direct fix (proposed, not run):** make the loss see relations. Image-side hard negatives by graph perturbation (the
  same image with one relation relabelled or reversed, as a negative for its own captions), the image-side counterpart of the
  caption swap negatives. It needs no new data and directly tests whether the relation generators become identified.

## 6. Files
- Code: `modules/models/vision/graph_tn.py`, `modules/models/text/einsum_classical.py` (data leaves, dense grads),
  `modules/utils/factory.py` (`vision.method: graph`), configs `configs/tn/s7_psg_{r0,graph}.yaml`.
- Tools: `tools/tn/s7_psg_graphs.py`, `s7_crops.py`, `s7_graph_check.py`, `s7_probe.py`, `s7_corrupt.py`, `s7_relstats.py`.
- Runs (tails checkpoints/mscoco/tn): tn_s7_{r0,g1,g2}_s{0,1}, tn_s7_g3_s0_hpc (the cluster run; tn_s7_g3_s0 is a pool attempt
  that crashed at start with OOM, kept). Per-error results: results/tn/step7/corrupt_*.json.

# Step 5: decomposition of the type tensors (report, 11 Nov 2026)

## 1. Answer
CP rank 128 on shared argument cores is the recommended decomposition: test rsum 274.8 ± 0.5 over 3 seeds with 9.0M text parameters, against 135.3 for the step-4c best (o16v3, 16.2M) and 8.8 for the step-1 reference (538M). It is the best model on recall, SVO-probes (0.805) and SugarCREPE++ replace-object (0.734), keeps word order (SVO swap_full 0.947), and is one of the cheapest to train (4.6 GB peak, 1040 samples/s on one H100). Bag-of-words is still higher on recall (rsum 305.3) but cannot tell word orders apart (swap_full 0.205).
- Ranking at matched parameters (~14M): CP 270 > tensor ring 258 > MPS 188. MPS needs 3x the parameters (bond 32, 44.8M) to reach 239.
- Rank: CP 64 263.5, 128 274.8, 256 269.7, 384 269.3, 512 270.1. Higher ranks peak earlier (best epoch ~57 at 384/512, 110-146 at 128) and do not help: the rank acts as a regulariser.
- Sharing the argument cores across complex types (S1, by leg type, side and distance from the word leg) costs nothing at equal bond and frees parameters for a larger bond (S1 bond 16: +39% rsum at fewer parameters than o16v3).
- Orthogonal init (init_orth) makes no difference for any family (MPS 187.6 vs 189.0 without, TR 257.6 vs 258.6, CP256 269.7 vs 268.8).
- The levelwise engine was rewritten (unique cores instead of per-use copies): training peak memory halves (o16v3 3.70 -> 1.95 GB, S1 bond 16 9.06 -> 4.46 GB) with identical losses.

Compositional targets are still far: ARO-Rel 0.525, ARO-Att 0.556, SC++ swap/relation subsets at or below chance, Winoground below chance. Section 6 says why; three follow-ups (E1-E3) are approved and E2 is running.

## 2. Setup
Base: o16v3 from step 4c (decoupled words, shared output core, output bond 16, centred base-point words, step-4 training setup: scale_free, no-bias head, constant LR 1e-3, 150 epochs, best.pt by val mrr). Test sets as in every step: MSCOCO 5k retrieval (rsum = R@1+5+10 over i2t and t2i, in %, max 600), ARO, SVO-probes, SugarCREPE++ (both positives must beat the negative, chance 1/3), Winoground. Seeds: 2 per configuration, 3 for the finalists (CP128, CP256); step 1: 3 seeds, step 4: 5. Branch tn-step5 (quantum-learning-labs/QCLIP), tables results/tn/step5/final2.

## 3. Results: every configuration against the baselines
| model | text params | rsum | i2t / t2i R@10 | ARO-Rel | ARO-Att | SVO std | swap_full | SC++ repl-obj | SC++ repl-rel | Wino text |
|---|---|---|---|---|---|---|---|---|---|---|
| step 1 reference (3) | 537.9M | 8.8 ± 1.0 | 0.041 / 0.009 | 0.494 | 0.488 | 0.509 | 0.497 | 0.375 | 0.341 | 0.145 |
| step 4 (5) | 18.0M | 109.2 ± 2.1 | 0.350 / 0.245 | 0.544 | 0.525 | 0.725 | 0.966 | 0.476 | 0.296 | 0.139 |
| step 4c o16v3 (2) | 16.2M | 135.3 ± 0.5 | 0.415 / 0.304 | 0.548 | 0.525 | 0.748 | 0.976 | 0.532 | 0.317 | 0.162 |
| MPS S1 bond 16 (2) | 14.1M | 187.6 ± 1.2 | 0.541 / 0.409 | 0.535 | 0.534 | 0.761 | 0.963 | 0.608 | 0.313 | 0.144 |
| MPS S1 bond 32 (2) | 44.8M | 238.6 ± 1.2 | 0.646 / 0.496 | 0.533 | 0.589 | 0.777 | 0.968 | 0.622 | 0.317 | 0.145 |
| tensor ring bond 16 (2) | 14.3M | 257.6 ± 1.3 | 0.687 / 0.520 | 0.526 | 0.571 | 0.789 | 0.957 | 0.668 | 0.332 | 0.138 |
| CP rank 64 (2) | 6.4M | 263.5 ± 1.9 | 0.700 / 0.524 | 0.535 | 0.522 | 0.785 | 0.949 | 0.737 | 0.364 | 0.141 |
| **CP rank 128 (3)** | **9.0M** | **274.8 ± 0.5** | **0.719 / 0.546** | 0.525 | 0.556 | **0.805** | 0.947 | 0.734 | 0.369 | 0.156 |
| CP rank 256 (3) | 14.3M | 269.7 ± 1.0 | 0.709 / 0.542 | 0.522 | 0.579 | 0.801 | 0.945 | 0.716 | 0.373 | 0.169 |
| CP rank 384 (2) | 19.5M | 269.3 ± 1.2 | 0.712 / 0.544 | 0.526 | 0.585 | 0.796 | 0.948 | 0.707 | 0.379 | 0.146 |
| CP rank 512 (2) | 24.8M | 270.1 ± 2.7 | 0.706 / 0.546 | 0.523 | 0.590 | 0.800 | 0.954 | 0.689 | 0.362 | 0.158 |
| bag-of-words (1) | 18.1M | 305.3 | 0.759 / 0.590 | 0.256 | 0.250 | 0.789 | 0.205 | 0.858 | 0.516 | 0.049 |

Spreads of the compositional scores are typically 0.004-0.03 (full mean ± std per benchmark in the appendix). Chance: ARO, SVO 0.5; SC++ 0.333; Winoground text 0.25. CLIP ViT-B/32's own text encoder on the same frozen image embeddings: rsum 359.6.

Reading: CP gains most where the rank meets the data, on recall and on replacing objects/attributes (SC++ replace-object 0.532 -> 0.734). ARO-Att rises with rank (0.52 at 64 -> 0.59 at 512) while recall does not: attribute binding wants more interaction capacity than retrieval does. ARO-Rel, the SC++ swap subsets and Winoground do not move with any decomposition.

## 4. Efficiency (controlled: one H100, same code, 300 training steps + one no-grad pass over 2,560 validation captions)
| model | text params | train peak (GB) | val no-grad peak (GB) | samples/s |
|---|---|---|---|---|
| o16v3 (step 4c) | 16.2M | 1.95 | 1.01 | 965 |
| MPS S1 bond 16 | 14.1M | 4.17 | 2.42 | 965 |
| tensor ring bond 16 | 14.3M | 10.55 | 3.26 | 718 |
| CP rank 64 | 6.4M | 1.80 | 1.33 | 1339 |
| **CP rank 128** | **9.0M** | 4.62 | 4.01 | 1042 |
| CP rank 256 | 14.3M | 5.68 | 2.72 | 1068 |

Training is launch-bound (text forward ~0.3 s/step of CPU dispatch), so samples/s varies little with size; tensor ring pays for its closing bond (2x the stored activations). Timings come from one node on our avoid list (seymour1, possibly shared): read them as relative. The step-1 reference trained at 12.7 min/epoch with 538M parameters; CP128 trains at ~1-2 min/epoch (cluster L4/L40S runs), 2.1 min average over its 3 runs.

Engine rewrite (same RTX 3090 Ti, identical losses to 6e-6 relative): the old levelwise engine copied every core once per use (o:out_mid 903 times = 0.44 GB). It now gathers unique rows, broadcasts single-row operands and frees pieces after their last use.
| | train peak old -> new (GB) | s/step old -> new |
|---|---|---|
| o16v3 | 3.70 -> 1.95 | 0.254 -> 0.229 |
| MPS S1 bond 16 | 9.06 -> 4.46 | 0.358 -> 0.254 |

Earlier in this step the centred pass of o16v3 was merged into one schedule: 0.871 -> 0.601 s/step (-31%), losses equal to 1.5e-7 relative.

## 5. How the decomposition was chosen
- 1. S screen (MPS bond 8, 2 seeds): S0 o16v3 135.3, S1 (share argument cores by leg type, side, distance) 135.0 at 46% of the parameters, S2 (no distance) 131.9 with less reliable word order. S1 bond 16: 187.6.
- 2. Families at ~14M parameters, with and without init_orth: CP rank 256 270.1, tensor ring bond 16 257.6, MPS bond 16 187.6, MPS bond 24 223.5 (26.9M), bond 32 238.6 (44.8M). Tucker was dropped: its core grows as r^K and orders reach 17.
- 3. CP rank sweep 64-512 (above): 128 wins; 3 seeds confirm (274.5, 275.4, 274.5).

## 6. Diagnostics: why the compositional scores do not move

Run on CP128 (2 seeds), CP256, o16v3 and BoW (results/tn/step5/audit).
- Benchmark artefacts ruled out: no TN produces ties; OOV words change Winoground text by 0.006 and SC++ by <= 0.01.
- ARO-Rel is 63% "to the left of / to the right of" (2,296 of 3,650 items): every model scores 0.50 there, so the pooled frozen CLIP B/32 image embedding very likely does not encode left/right. 210 more items use symmetric relations (next to, near) and cannot be answered. On the remaining 1,096 items CP128 reaches 0.58-0.59.
- Relation words are ignored. Swapping one word vector and measuring 1 - cos of the caption embedding (COCO captions, CP128): nouns 0.16, adjectives 0.08, determiners 0.04, prepositions 0.013 (69% of occurrences < 0.01), 3-leg relation words (of, with, and) 0.010. Decoupling gives all prepositions one shared type tensor, so "on" and "under" differ only by a word vector, and training learns to ignore it: the structure is used (swap_full 0.95) but the identity of the relation is not.
- Fragmentation of meaning across grammatical roles: one word's uses spread over many type tensors ("on": 69 signatures, 3 main roles; root and non-root uses of the same category are separate tensors). Items whose rarest type tensor has <= 1,000 training uses score 0.09-0.21 lower on every SC++ subset, also at matched caption length.

## 7. Approved follow-ups (in progress)
- E2, Frobenius lifting (running): one core per word; its higher-order roles (VP-modifying prepositions, copulas, relative pronouns, coordination, adverbs) are built from that core with parameter-free copy/merge spiders, and every caption leaves through one codomain map per atomic type. Rules checked on the training wiring (dominant pattern 86-100%); 10.6% of training word tokens lifted; type tensors 144 -> 96. Tests: compile unchanged with lifting off; 0 invalid networks; engine = direct contraction (cos 1.000). Runs: CP128 + full lifting and + root independence only, 2 seeds each.
- E3: Winoground caption-bias diagnostic (how often both images prefer the same caption).
- E1: own tensors for closed-class relation words, on top of E2.
- Cheap-fix screen on CP128 (weight decay, batch 1024, 300 epochs, word dropout, EMA; CLIP word init as a temporary stand-in): code done and checked (default losses bit-identical).

## 8. Worth noting
- Contrastive training makes even an explicitly compositional model a "bag of nouns": the grammar wires the relation words in, the objective drains them (Section 6). This mirrors the ARO paper's finding for CLIP, inside a model whose architecture forces composition. E1/E2 test whether capacity or structure changes it; if neither does, the cause is the objective.
- CP rank works as a regulariser for shared generators (128 > 256 > 512, higher ranks overfit earlier). The effective rank each type tensor uses would say which grammatical relations need rich interactions.
- E2 turns parse trees into graphs with cycles (copy and merge share wires), a grammar-generated family just beyond trees: the treewidth of lifted networks is a direct testbed for the viva's M2 / prediction MH. In a quantum realisation copy is an isometry but merge needs post-selection, so each merge spends success probability (M1).

## 9. Not done / held back
- Tucker (core size r^K at orders up to 17). o32 + centring at a longer budget: superseded by the CP results.
- Speed timings are from a node on the avoid list (relative only); the pool had no free GPU for a clean 3090 Ti profile of every family.
- Bag-of-words has 1 seed; CP384/512 and the families 2 seeds.

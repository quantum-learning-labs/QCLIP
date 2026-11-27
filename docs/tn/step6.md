# Step 6: token sub-networks for words (report, 26 Nov 2026)

## 1. Answer
- **Token sub-networks change nothing at test.** Each word becomes a chain of CLIP-BPE pieces, w = M(tn)…M(t2) v(t1) with
  M(t) = A diag(m_t) B and identity init.
  - rsum 326.3 (325.6 / 326.9) vs 325.2 for the step-5 TN.
  - ARO-Att 0.718, ARO-Rel 0.573, SVO 0.843, swap_full 0.996: all within seed noise of step 5.
- **It does reduce memorisation, but by fitting training images less, not by generalising better.** Train-5k rsum 403 vs 412;
  the train-test gap falls from 86 to 77 while test is unchanged.
- **Out-of-vocabulary captions do not improve.** The 562 test captions that contain a word never seen in training as a whole
  word score t2i R@10 0.472 with tokens vs 0.491 without; BoW gets 0.596.
  - Test sentences with an untrained symbol halve (OOV sentence fraction 9.7% → 4.5%).
  - But being made of known pieces does not make a rare word usable. BPE pieces of rare COCO words ('sur'+'boar'+'ders',
    'cloud'+'wreath'+'ed') share spelling, not meaning, and a contrastive loss on COCO alone gives them no reason to compose.
- **Retracted (step 6n below): "multiplicative composition is not robust to a single bad leaf".** The unseen-word bucket
  drop (TN 0.18-0.20 vs BoW 0.14 R@10) is real, but direct tests show it is not leaf-error propagation: one corrupted word
  costs the TN the same as BoW (section 5).
- **Isometric generators (the "quantum-implementable" row) cost a lot, but the comparison is confounded.** T2 = MPS bond 16
  with left-canonical (isometric) cores + tokens: rsum 258.8 (−67), ARO-Att 0.653, ARO-Rel 0.555; Winoground text 0.195 (the
  highest TN value, n = 390, about 1.5 SE above T1).
  - T2 changes two things at once (MPS vs CP, isometric vs free). Step 4 already found MPS below CP under the old recipe.
  - Separating the two needs one control: free MPS bond 16 + tokens.

## 2. Setup
Code tn-step6: 0a21ca0 (text.tokens), checks `tools/tn/s6_tok_check.py`:
- compile: 0 mismatches;
- engine vs per-caption reference: 5.6e-7;
- identity chain vs root vector: 7.8e-8;
- tokens off: bit-identical losses vs 783c288.

Recipe = step 5 (CP128 + R1, learnable temperature + v1 negatives, EMA 0.999, 150 epochs, best by val MRR).
- 1.8% of word uses are multi-piece (rare words); 19,349 distinct pieces.
- Unseen test word types: 557, of which 367 are fully built from trained pieces.
- Params 3.47M (CP, +1% vs step 5); same memory and speed.

## 3. Results (test)
| model | seeds | rsum | train5k | ARO-Att | ARO-Rel | SVO | swap_full | SC++ repl obj/att/rel | SC++ swap obj/att | Wino text/image |
|---|---|---|---|---|---|---|---|---|---|---|
| step-5 TN (5d) | 2 | 325.2 | 412.3 (s0) | 0.725 | 0.576 | 0.845 | 0.996 | 0.747 / 0.503 / 0.448 | 0.371 / 0.425 | 0.169 / 0.077 |
| **T1 tokens** | 2 | 326.3 | 403.3 | 0.718 | 0.573 | 0.843 | 0.996 | 0.733 / 0.496 / 0.438 | 0.396 / 0.430 | 0.148 / 0.081 |
| T2 tokens + isometric MPS b16 | 1 | 258.8 | 353.4 | 0.653 | 0.555 | 0.824 | 0.988 | 0.643 / 0.364 / 0.389 | 0.336 / 0.365 | 0.195 / 0.092 |
| BoW, same words (5e) | 2 | 370.5 | 451.2 (s0) | 0.263 | 0.273 | 0.846 | 0.21 | 0.865 / 0.690 / 0.594 | 0.150 / 0.246 | 0.060 / 0.015 |

**Test t2i R@10 by the caption's words (word-level buckets from the step-5 compile; TN values = 2-seed means):**

| captions | n | step-5 TN | T1 tokens | BoW |
|---|---|---|---|---|
| contain a word unseen in training | 562 | 0.491 | 0.472 | 0.596 |
| rarest word seen 1-9 times | 1463 | 0.602 | 0.602 | 0.684 |
| all others | 22884 | 0.671 | 0.674 | 0.735 |

Results: results/tn/step6/ (benchmarks, breakdowns, gaps; worktree, untracked). Two pool runs ended without their post-training
benchmark (cause unknown); best.pt was benchmarked afterwards with the same script.

## 4. Next
Step 6 closes without a recall gain. The lexical lever is not sharing by spelling. Step 6n (section 5) shows the TN is as
robust to one bad leaf as BoW, so step 7 does not need a robustness fix first; measuring the cost of each wrong node or
relation (GT graphs A vs generator graphs B) is still worth reporting.

## 5. Step 6n: robustness to one bad leaf (eval only, branch tn-neutral)
Question: does an error in one leaf propagate through the multilinear tree (the product-of-similarities argument: one leaf
with similarity 0 to the right word would zero the sentence similarity), which would call for a neutral-element mixture in
the composition rule? Two tests on the step-5 TN (tn_s5d_ltneg_s{0,1}), the token TN and BoW (tn_s5e_bow_s{0,1}); no training.

**Test 1, unseen words set to the neutral leaf** (`text.oov_init`, 771dc14: random = seeded init as before, neutral = bank
row 0 so the word is word_base alone, mean = mean trained row). t2i R@10 on the 562 test captions with a word unseen in
training (rest 0.667 TN, 0.670 token TN, 0.732 BoW):

| model | random | neutral | mean |
|---|---|---|---|
| TN (step 5) | 0.491 | 0.496 | 0.496 |
| token TN | 0.472 | 0.478 | 0.476 |
| BoW | 0.603 | 0.615 | 0.605 (1 seed) |

COCO rsum moves by < 0.3 for every model. The untrained vectors were not doing damage; the unseen-word gap is missing
information, not a bad leaf. (Untrained symbols at test are 2.2% of captions, 544 'w:' + 2 't:'; the benchmark's
oov_sentence_frac 9.7% counts compiled sentence arrays differently.)

**Test 2, one corrupted word per caption** (`tools/tn/s6n_corrupt.py`): in every test caption one word leaf, chosen uniformly
(seeded, the same leaf in both modes), is replaced by the neutral leaf or by a trained word drawn by training frequency
(wrong but valid, like a detector error). t2i R@10, mean of 2 seeds:

| model | clean | neutral | swap |
|---|---|---|---|
| TN (step 5) | 0.663 | −0.059 (8.9%) | −0.088 (13.2%) |
| BoW | 0.729 | −0.056 (7.6%) | −0.091 (12.5%) |

Swap drop by caption length (words: <9 / 9-10 / 11-12 / 13+; n = 4281 / 10874 / 6253 / 3501): TN 0.107 / 0.093 / 0.076 /
0.066, BoW 0.115 / 0.098 / 0.078 / 0.061. Both fall with length at the same rate: the TN dilutes one bad word like an average.

**Conclusion.** The TN is not more fragile than BoW to one bad leaf, so the product-of-similarities picture does not describe
the trained model. Likely reason (hypothesis, not tested here): since step 4c every word is word_base + d_w, so each leaf
already has a shared neutral component, and the expansion of the tree over (base + d) contains the low-order, BoW-like terms
that a neutral-element mixture would add. The proposed per-leaf confidence mixture is therefore not pursued. For step 7: give
image-graph nodes the same base + specific form, and keep the per-error cost as a measured quantity.
Results: results/tn/step6n/ (breakdown_*_{neutral,mean}.json, corrupt_*.json), benchmark_best_oov_{neutral,mean}.json per run.

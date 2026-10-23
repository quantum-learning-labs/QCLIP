# Step 2 tables (generated: tools/tn/step2_table.py, tools/tn/summarise.py)

## Per change (pinned 3090 Ti, seed 0, epoch 0 = 222 steps; eval rows/s on tn_s1_clip_s0/last.pt)

Startup = build + compile + text_init (incl. plan search) + to_device + loader/optimizer. 'baseline (unpinned)' ran addfbe3 code (its JSON records the HEAD at write time).

| change | train samples/s | vs baseline | vs previous | step ms: fwd text / backward / optimizer / data | startup s | peak GB | coco5k rows/s | sugarcrepe_replace_object rows/s | winoground rows/s | aro_rel rows/s | regression |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (addfbe3, unpinned) (3ccfabb) | 225 | 1.00x | - | 790 / 1277 / 102 / 46 | 41 | 4.1 | - | - | - | - |  |
| baseline pinned (0889364) | 241 | 1.07x | 1.07x | 731 / 1204 / 103 / 46 | 61 | 4.1 | 161 | 214 | 462 | 3645 | loss 222/222 identical, max 0.0e+00; metrics 88/88 identical; scores max 0.0e+00 |
| 1 benchmark vocab from ckpt (+compile cache, dropped) (3ccfabb) | 242 | 1.07x | 1.01x | 720 / 1204 / 103 / 44 | 60 | 4.1 | 160 | 218 | 460 | 3685 | loss 222/222 identical, max 0.0e+00; metrics 88/88 identical; scores max 0.0e+00 |
| 2+4 cached plans, one output gather (4326aa2) | 247 | 1.10x | 1.02x | 678 / 1194 / 86 / 67 | 58 | 4.1 | 168 | 238 | 486 | 5600 | loss 222/222 identical, max 0.0e+00; metrics 88/88 identical; scores max 0.0e+00 |
| 3 banks + RowAdam (c482aeb) | 311 | 1.38x | 1.26x | 566 / 955 / 10 / 67 | 54 | 6.5 | 184 | 233 | 537 | 6575 | loss 187/222 identical, max 4.8e-07; metrics 88/88 identical; scores max 0.0e+00 |
| 6 levelwise (path order) (e0fcf45) | 566 | 2.51x | 1.82x | 278 / 561 / 11 / 43 | 54 | 6.7 | 321 | 474 | 804 | 6805 | loss 173/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 8.2e-08 |
| 7 one gather per round/shape (a2cf252) | 984 | 4.37x | 1.74x | 223 / 220 / 6 / 65 | 35 | 6.9 | 331 | 515 | 919 | 6765 | loss 171/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 8.2e-08 |
| 8 paths up front, no pin_memory (212227b) | 1179 | 5.23x | 1.20x | 192 / 218 / 5 / 12 | 44 | 6.9 | 383 | 579 | 999 | 15016 | loss 171/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 8.2e-08 |
| 9 depth scheduling + positional keys (44f7f4f) | 3543 | 15.72x | 3.01x | 96 / 31 / 4 / 12 | 44 | 6.8 | 441 | 601 | 1706 | 19420 | loss 174/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 6.7e-08 |
| 10 plans in pool (0e9b66c) | 4809 | 21.34x | 1.36x | 57 / 32 / 4 / 12 | 51 | 6.8 | 433 | 599 | 1650 | 19713 | loss 174/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 6.7e-08 |
| 11 eval plans up front, vector mask (349d7d8) | 3891 | 17.26x | 0.81x | 68 / 37 / 9 / 13 | 60 | 6.8 | 788 | 2395 | 3897 | 18531 | loss 174/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 6.7e-08 |
| 12 serial search for small sets (eb67fc7) | 4839 | 21.47x | 1.24x | 57 / 31 / 4 / 12 | 51 | 6.8 | 666 | 2120 | 3816 | 18818 | loss 174/222 identical, max 9.5e-07; metrics 84/88 identical; scores max 6.7e-08 |

## Step 1 vs step 2, seeds 0-2 (step 1: tn_s1_clip_h_s*, HPC L40S/RTX 6000 Ada; step 2: tn_s2_opt_s*, 3090 Ti, levelwise)

| benchmark | chance | n | step1 mean ± std (seeds) | step2 mean ± std (seeds) | step1 above chance (z) | step2 above chance (z) | OOV sent. |
|---|---|---|---|---|---|---|---|
| MSCOCO 5k i2t R@1 | 0.0002 | 5000 | 0.0065 ± 0.0008 (3) | 0.0071 ± 0.0011 (3) | +0.0063 (5.6) | +0.0069 (5.8) | 0.188 |
| MSCOCO 5k i2t R@5 | 0.0010 | 5000 | 0.0251 ± 0.0039 (3) | 0.0250 ± 0.0043 (3) | +0.0241 (10.9) | +0.0240 (10.9) | 0.188 |
| MSCOCO 5k i2t R@10 | 0.0020 | 5000 | 0.0414 ± 0.0040 (3) | 0.0407 ± 0.0039 (3) | +0.0394 (14.0) | +0.0387 (13.8) | 0.188 |
| MSCOCO 5k t2i R@1 | 0.0002 | 5000 | 0.0011 ± 0.0002 (3) | 0.0011 ± 0.0002 (3) | +0.0009 (1.9) | +0.0009 (1.9) | 0.188 |
| MSCOCO 5k t2i R@5 | 0.0010 | 5000 | 0.0047 ± 0.0006 (3) | 0.0047 ± 0.0006 (3) | +0.0037 (3.8) | +0.0037 (3.8) | 0.188 |
| MSCOCO 5k t2i R@10 | 0.0020 | 5000 | 0.0088 ± 0.0008 (3) | 0.0088 ± 0.0008 (3) | +0.0068 (5.1) | +0.0068 (5.1) | 0.188 |
| ARO VG-Relation | 0.5000 | 3650 | 0.4944 ± 0.0066 (3) | 0.4946 ± 0.0064 (3) | -0.0056 (-0.7) | -0.0054 (-0.7) | 0.000 |
| ARO VG-Attribution | 0.5000 | 4438 | 0.4879 ± 0.0143 (3) | 0.4879 ± 0.0143 (3) | -0.0121 (-1.6) | -0.0121 (-1.6) | 0.002 |
| SVO-probes std (image choice) | 0.5000 | 1833 | 0.5090 ± 0.0185 (3) | 0.5090 ± 0.0185 (3) | +0.0090 (0.8) | +0.0090 (0.8) | 0.000 |
| SVO swap_full (text choice) | 0.5000 | 1777 | 0.4975 ± 0.0105 (3) | 0.4971 ± 0.0104 (3) | -0.0025 (-0.2) | -0.0029 (-0.2) | 0.000 |
| SVO swap_animate (text choice) | 0.5000 | 55 | 0.5697 ± 0.0378 (3) | 0.5697 ± 0.0378 (3) | +0.0697 (1.0) | +0.0697 (1.0) | 0.000 |
| SugarCREPE++ swap_object (both pos > neg) | 0.3333 | 244 | 0.3333 ± 0.0434 (3) | 0.3361 ± 0.0376 (3) | +0.0000 (0.0) | +0.0027 (0.1) | 0.053 |
| SugarCREPE++ swap_atribute (both pos > neg) | 0.3333 | 661 | 0.3666 ± 0.0152 (3) | 0.3656 ± 0.0174 (3) | +0.0333 (1.8) | +0.0323 (1.7) | 0.040 |
| SugarCREPE++ replace_object (both pos > neg) | 0.3333 | 1643 | 0.3753 ± 0.0225 (3) | 0.3770 ± 0.0225 (3) | +0.0420 (3.5) | +0.0436 (3.6) | 0.049 |
| SugarCREPE++ replace_attribute (both pos > neg) | 0.3333 | 784 | 0.3631 ± 0.0070 (3) | 0.3614 ± 0.0039 (3) | +0.0298 (1.7) | +0.0281 (1.6) | 0.048 |
| SugarCREPE++ replace_relation (both pos > neg) | 0.3333 | 1393 | 0.3408 ± 0.0109 (3) | 0.3405 ± 0.0112 (3) | +0.0074 (0.6) | +0.0072 (0.6) | 0.050 |
| Winoground text | 0.2500 | 390 | 0.1453 ± 0.0146 (3) | 0.1479 ± 0.0141 (3) | -0.1047 (-5.9) | -0.1021 (-5.7) | 0.236 |
| Winoground image | 0.2500 | 390 | 0.1513 ± 0.0136 (3) | 0.1513 ± 0.0136 (3) | -0.0987 (-5.4) | -0.0987 (-5.4) | 0.236 |
| Winoground group | 0.1667 | 390 | 0.0641 ± 0.0000 (3) | 0.0650 ± 0.0015 (3) | -0.1026 (-8.3) | -0.1017 (-8.2) | 0.236 |

| efficiency | step1 | step2 |
|---|---|---|
| params_text | 537,860,382 | 537,860,382 |
| params_image_trainable | 0 | 0 |
| train_samples_per_s | 150.97 | 4,759.48 |
| epoch_train_min | 12.67 | 0.40 |
| val_eval_s | 67.61 | 4.81 |
| peak_train_mem_gb | 5.11 | 8.04 |
| gpus | ['NVIDIA L40S', 'NVIDIA RTX 6000 Ada Generation'] | ['NVIDIA GeForce RTX 3090 Ti'] |

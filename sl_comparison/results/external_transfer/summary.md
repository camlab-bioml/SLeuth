# External-screen transfer (zero-shot + fine-tuning)

PCA arm: none (no PCA)

Metric: Spearman(score, target), target = -1 x screen phenotype (higher = more SL). Cells are mean ± population sd (ddof=0) over the training folds; per-fold values are in the CSVs and in the per-fold section below.

Screen splits: cv1 = random pairs, cv2 = >=1 gene unseen, cv3 = both genes unseen.

**The random / >=1-unseen / both-unseen columns only bear on fine-tuning.** They say how the SCREEN's pairs were divided into train / val / test (eval_finetuning --cv_type), i.e. which test genes the fine-tuned model saw in the screen's train split. A zero-shot score never uses the screen's train rows, so across the zero-shot columns only the test-row subset changes, not what the model was trained on.

Matched rows = test, not in SynLethDB (SL or screened non-SL), both genes in genes.txt. Every model below is scored on exactly these rows. SLeuth scores are RAW (uncalibrated) per-fold logits.

## Zero-shot, matched rows

### Adamson

| Model | random pairs (n=14,727) | >=1 gene unseen (n=25,782) | both genes unseen (n=2,709) |
|---|---|---|---|
| siamese_sl_pd_pooled | -0.0843 ± 0.0496 | -0.0789 ± 0.0648 | -0.0882 ± 0.0932 |
| siamese_sl | -0.2069 ± 0.0348 | -0.2106 ± 0.0313 | -0.2385 ± 0.0458 |
| SLMGAE | -0.0310 ± 0.0297 | -0.0318 ± 0.0373 | -0.0384 ± 0.0478 |
| SLGNN | -0.1143 ± 0.0208 | -0.1418 ± 0.0313 | -0.1990 ± 0.0543 |
| NSF4SL | +0.0638 ± 0.1370 | +0.1030 ± 0.1130 | +0.1266 ± 0.0800 |
| DegreePrior | +0.2124 ± 0.0381 | +0.2497 ± 0.0662 | +0.2951 ± 0.1280 |

### AdamsonGI

| Model | random pairs (n=14,586) | >=1 gene unseen (n=27,404) | both genes unseen (n=3,181) |
|---|---|---|---|
| siamese_sl_pd_pooled | +0.0623 ± 0.0196 | +0.0661 ± 0.0298 | +0.0736 ± 0.0470 |
| siamese_sl | +0.0644 ± 0.0176 | +0.0668 ± 0.0239 | +0.0767 ± 0.0321 |
| SLMGAE | +0.0219 ± 0.0089 | +0.0418 ± 0.0144 | +0.0767 ± 0.0263 |
| SLGNN | +0.0707 ± 0.0172 | +0.0595 ± 0.0109 | +0.0636 ± 0.0158 |
| NSF4SL | +0.0072 ± 0.0170 | +0.0057 ± 0.0161 | +0.0185 ± 0.0239 |
| DegreePrior | +0.0174 ± 0.0135 | +0.0054 ± 0.0099 | -0.0073 ± 0.0271 |

### Corn

| Model | random pairs (n=16,614) | >=1 gene unseen (n=29,200) | both genes unseen (n=3,117) |
|---|---|---|---|
| siamese_sl_pd_pooled | +0.0409 ± 0.0167 | +0.0471 ± 0.0196 | +0.0660 ± 0.0263 |
| siamese_sl | +0.0465 ± 0.0113 | +0.0521 ± 0.0111 | +0.0538 ± 0.0198 |
| SLMGAE | +0.0434 ± 0.0162 | +0.0515 ± 0.0214 | +0.0787 ± 0.0216 |
| SLGNN | +0.0345 ± 0.0116 | +0.0175 ± 0.0135 | +0.0157 ± 0.0245 |
| NSF4SL | +0.0135 ± 0.0183 | -0.0066 ± 0.0292 | -0.0390 ± 0.0385 |
| DegreePrior | +0.0526 ± 0.0070 | +0.0713 ± 0.0052 | +0.0981 ± 0.0172 |

### Gilbert

| Model | random pairs (n=1,383) | >=1 gene unseen (n=2,590) | both genes unseen (n=299) |
|---|---|---|---|
| siamese_sl_pd_pooled | +0.0162 ± 0.0334 | +0.0379 ± 0.0429 | +0.1006 ± 0.0462 |
| siamese_sl | +0.0220 ± 0.0187 | +0.0313 ± 0.0343 | +0.0783 ± 0.0627 |
| SLMGAE | -0.0238 ± 0.0171 | -0.0141 ± 0.0330 | +0.0272 ± 0.0484 |
| SLGNN | -0.0263 ± 0.0156 | +0.0054 ± 0.0185 | +0.0532 ± 0.0464 |
| NSF4SL | -0.0068 ± 0.0142 | -0.0011 ± 0.0351 | +0.0767 ± 0.0728 |
| DegreePrior | -0.0261 ± 0.0148 | +0.0078 ± 0.0279 | +0.0480 ± 0.0634 |

## Screen role (SynLethDB overlap)

A screen at least 50% of whose scored pairs SynLethDB lists (SL or screened non-SL) is `in_distribution_check`: its pairs largely are the models' training data, so it checks in-distribution agreement, not transfer; otherwise `external`. `sldb_overlap.csv` has the counts.

| Screen | screen_role | pairs | SynLethDB SL | screened non-SL | either | in fold training (mean) |
|---|---|---|---|---|---|---|
| Adamson | external | 146,583 | 0.004 | 0.029 | 0.033 | 0.017 |
| AdamsonGI | external | 143,354 | 0.004 | 0.029 | 0.033 | 0.017 |
| Corn | external | 149,787 | 0.002 | 0.007 | 0.008 | 0.004 |
| Gilbert | in_distribution_check | 99,681 | 0.060 | 0.814 | 0.874 | 0.455 |

## Zero-shot, all matched pairs vs held out

`matched_incl_sldb` = test, both genes in genes.txt, SynLethDB pairs kept; `matched` = the held-out part (not in SynLethDB); `matched_in_sldb` = the SynLethDB part, where `SynLethDB_label` (1 = SL, 0 = screened non-SL) is scored as a predictor too.

| Model | Screen | Row set | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|---|
| siamese_sl_pd_pooled | Adamson | matched | -0.0843 ± 0.0496 (n=14,727) | -0.0789 ± 0.0648 (n=25,782) | -0.0882 ± 0.0932 (n=2,709) |
| siamese_sl_pd_pooled | Adamson | matched_incl_sldb | -0.1023 ± 0.0520 (n=15,669) | -0.0970 ± 0.0658 (n=27,317) | -0.1030 ± 0.0917 (n=2,850) |
| siamese_sl_pd_pooled | Adamson | matched_in_sldb | -0.1258 ± 0.0462 (n=942) | -0.0521 ± 0.0736 (n=1,535) | +0.0604 ± 0.2189 (n=141) |
| siamese_sl_pd_pooled | AdamsonGI | matched | +0.0623 ± 0.0196 (n=14,586) | +0.0661 ± 0.0298 (n=27,404) | +0.0736 ± 0.0470 (n=3,181) |
| siamese_sl_pd_pooled | AdamsonGI | matched_incl_sldb | +0.0648 ± 0.0179 (n=15,527) | +0.0684 ± 0.0298 (n=28,900) | +0.0760 ± 0.0463 (n=3,321) |
| siamese_sl_pd_pooled | AdamsonGI | matched_in_sldb | +0.0825 ± 0.0320 (n=941) | +0.0972 ± 0.0425 (n=1,496) | +0.1062 ± 0.0615 (n=140) |
| siamese_sl_pd_pooled | Corn | matched | +0.0409 ± 0.0167 (n=16,614) | +0.0471 ± 0.0196 (n=29,200) | +0.0660 ± 0.0263 (n=3,117) |
| siamese_sl_pd_pooled | Corn | matched_incl_sldb | +0.0406 ± 0.0164 (n=16,864) | +0.0466 ± 0.0195 (n=29,624) | +0.0659 ± 0.0254 (n=3,160) |
| siamese_sl_pd_pooled | Corn | matched_in_sldb | +0.0125 ± 0.0427 (n=250) | +0.0027 ± 0.0585 (n=424) | -0.0275 ± 0.1218 (n=43) |
| siamese_sl_pd_pooled | Gilbert | matched | +0.0162 ± 0.0334 (n=1,383) | +0.0379 ± 0.0429 (n=2,590) | +0.1006 ± 0.0462 (n=299) |
| siamese_sl_pd_pooled | Gilbert | matched_incl_sldb | +0.1234 ± 0.0216 (n=18,786) | +0.1231 ± 0.0325 (n=32,886) | +0.1345 ± 0.0460 (n=3,486) |
| siamese_sl_pd_pooled | Gilbert | matched_in_sldb | +0.1387 ± 0.0217 (n=17,403) | +0.1323 ± 0.0360 (n=30,296) | +0.1357 ± 0.0511 (n=3,187) |
| siamese_sl | Adamson | matched | -0.2069 ± 0.0348 (n=14,727) | -0.2106 ± 0.0313 (n=25,782) | -0.2385 ± 0.0458 (n=2,709) |
| siamese_sl | Adamson | matched_incl_sldb | -0.2189 ± 0.0388 (n=15,669) | -0.2183 ± 0.0366 (n=27,317) | -0.2402 ± 0.0548 (n=2,850) |
| siamese_sl | Adamson | matched_in_sldb | -0.1338 ± 0.0600 (n=942) | -0.0135 ± 0.0832 (n=1,535) | +0.1057 ± 0.2134 (n=141) |
| siamese_sl | AdamsonGI | matched | +0.0644 ± 0.0176 (n=14,586) | +0.0668 ± 0.0239 (n=27,404) | +0.0767 ± 0.0321 (n=3,181) |
| siamese_sl | AdamsonGI | matched_incl_sldb | +0.0697 ± 0.0177 (n=15,527) | +0.0732 ± 0.0249 (n=28,900) | +0.0877 ± 0.0339 (n=3,321) |
| siamese_sl | AdamsonGI | matched_in_sldb | +0.1168 ± 0.0396 (n=941) | +0.1441 ± 0.0404 (n=1,496) | +0.2270 ± 0.0570 (n=140) |
| siamese_sl | Corn | matched | +0.0465 ± 0.0113 (n=16,614) | +0.0521 ± 0.0111 (n=29,200) | +0.0538 ± 0.0198 (n=3,117) |
| siamese_sl | Corn | matched_incl_sldb | +0.0473 ± 0.0110 (n=16,864) | +0.0530 ± 0.0109 (n=29,624) | +0.0543 ± 0.0182 (n=3,160) |
| siamese_sl | Corn | matched_in_sldb | +0.0170 ± 0.0449 (n=250) | -0.0334 ± 0.0298 (n=424) | -0.0931 ± 0.1734 (n=43) |
| siamese_sl | Gilbert | matched | +0.0220 ± 0.0187 (n=1,383) | +0.0313 ± 0.0343 (n=2,590) | +0.0783 ± 0.0627 (n=299) |
| siamese_sl | Gilbert | matched_incl_sldb | +0.1135 ± 0.0118 (n=18,786) | +0.1135 ± 0.0162 (n=32,886) | +0.1039 ± 0.0245 (n=3,486) |
| siamese_sl | Gilbert | matched_in_sldb | +0.1287 ± 0.0117 (n=17,403) | +0.1229 ± 0.0178 (n=30,296) | +0.1041 ± 0.0272 (n=3,187) |
| SLMGAE | Adamson | matched | -0.0310 ± 0.0297 (n=14,727) | -0.0318 ± 0.0373 (n=25,782) | -0.0384 ± 0.0478 (n=2,709) |
| SLMGAE | Adamson | matched_incl_sldb | -0.0307 ± 0.0344 (n=15,669) | -0.0333 ± 0.0439 (n=27,317) | -0.0440 ± 0.0555 (n=2,850) |
| SLMGAE | Adamson | matched_in_sldb | -0.1330 ± 0.0917 (n=942) | -0.0962 ± 0.1339 (n=1,535) | -0.0964 ± 0.2097 (n=141) |
| SLMGAE | AdamsonGI | matched | +0.0219 ± 0.0089 (n=14,586) | +0.0418 ± 0.0144 (n=27,404) | +0.0767 ± 0.0263 (n=3,181) |
| SLMGAE | AdamsonGI | matched_incl_sldb | +0.0227 ± 0.0121 (n=15,527) | +0.0439 ± 0.0172 (n=28,900) | +0.0804 ± 0.0284 (n=3,321) |
| SLMGAE | AdamsonGI | matched_in_sldb | +0.0510 ± 0.0618 (n=941) | +0.0932 ± 0.0692 (n=1,496) | +0.1418 ± 0.0507 (n=140) |
| SLMGAE | Corn | matched | +0.0434 ± 0.0162 (n=16,614) | +0.0515 ± 0.0214 (n=29,200) | +0.0787 ± 0.0216 (n=3,117) |
| SLMGAE | Corn | matched_incl_sldb | +0.0463 ± 0.0159 (n=16,864) | +0.0542 ± 0.0207 (n=29,624) | +0.0818 ± 0.0205 (n=3,160) |
| SLMGAE | Corn | matched_in_sldb | +0.0998 ± 0.0515 (n=250) | +0.0692 ± 0.0320 (n=424) | +0.0697 ± 0.1056 (n=43) |
| SLMGAE | Gilbert | matched | -0.0238 ± 0.0171 (n=1,383) | -0.0141 ± 0.0330 (n=2,590) | +0.0272 ± 0.0484 (n=299) |
| SLMGAE | Gilbert | matched_incl_sldb | +0.0623 ± 0.0258 (n=18,786) | +0.0679 ± 0.0303 (n=32,886) | +0.0766 ± 0.0388 (n=3,486) |
| SLMGAE | Gilbert | matched_in_sldb | +0.0710 ± 0.0291 (n=17,403) | +0.0758 ± 0.0323 (n=30,296) | +0.0811 ± 0.0381 (n=3,187) |
| SLGNN | Adamson | matched | -0.1143 ± 0.0208 (n=14,727) | -0.1418 ± 0.0313 (n=25,782) | -0.1990 ± 0.0543 (n=2,709) |
| SLGNN | Adamson | matched_incl_sldb | -0.1380 ± 0.0175 (n=15,669) | -0.1662 ± 0.0259 (n=27,317) | -0.2182 ± 0.0477 (n=2,850) |
| SLGNN | Adamson | matched_in_sldb | -0.0624 ± 0.0689 (n=942) | -0.0714 ± 0.0730 (n=1,535) | -0.0420 ± 0.1358 (n=141) |
| SLGNN | AdamsonGI | matched | +0.0707 ± 0.0172 (n=14,586) | +0.0595 ± 0.0109 (n=27,404) | +0.0636 ± 0.0158 (n=3,181) |
| SLGNN | AdamsonGI | matched_incl_sldb | +0.0721 ± 0.0146 (n=15,527) | +0.0611 ± 0.0100 (n=28,900) | +0.0662 ± 0.0168 (n=3,321) |
| SLGNN | AdamsonGI | matched_in_sldb | +0.0768 ± 0.0186 (n=941) | +0.0711 ± 0.0261 (n=1,496) | +0.0945 ± 0.0445 (n=140) |
| SLGNN | Corn | matched | +0.0345 ± 0.0116 (n=16,614) | +0.0175 ± 0.0135 (n=29,200) | +0.0157 ± 0.0245 (n=3,117) |
| SLGNN | Corn | matched_incl_sldb | +0.0340 ± 0.0116 (n=16,864) | +0.0160 ± 0.0136 (n=29,624) | +0.0154 ± 0.0239 (n=3,160) |
| SLGNN | Corn | matched_in_sldb | +0.0334 ± 0.0653 (n=250) | -0.0221 ± 0.0387 (n=424) | -0.0570 ± 0.1197 (n=43) |
| SLGNN | Gilbert | matched | -0.0263 ± 0.0156 (n=1,383) | +0.0054 ± 0.0185 (n=2,590) | +0.0532 ± 0.0464 (n=299) |
| SLGNN | Gilbert | matched_incl_sldb | +0.0657 ± 0.0235 (n=18,786) | +0.0666 ± 0.0167 (n=32,886) | +0.0737 ± 0.0207 (n=3,486) |
| SLGNN | Gilbert | matched_in_sldb | +0.0769 ± 0.0251 (n=17,403) | +0.0715 ± 0.0180 (n=30,296) | +0.0703 ± 0.0217 (n=3,187) |
| NSF4SL | Adamson | matched | +0.0638 ± 0.1370 (n=14,727) | +0.1030 ± 0.1130 (n=25,782) | +0.1266 ± 0.0800 (n=2,709) |
| NSF4SL | Adamson | matched_incl_sldb | +0.1057 ± 0.1309 (n=15,669) | +0.1438 ± 0.1085 (n=27,317) | +0.1614 ± 0.0766 (n=2,850) |
| NSF4SL | Adamson | matched_in_sldb | +0.0837 ± 0.0941 (n=942) | +0.0245 ± 0.0769 (n=1,535) | -0.0104 ± 0.1569 (n=141) |
| NSF4SL | AdamsonGI | matched | +0.0072 ± 0.0170 (n=14,586) | +0.0057 ± 0.0161 (n=27,404) | +0.0185 ± 0.0239 (n=3,181) |
| NSF4SL | AdamsonGI | matched_incl_sldb | -0.0005 ± 0.0153 (n=15,527) | -0.0000 ± 0.0135 (n=28,900) | +0.0151 ± 0.0206 (n=3,321) |
| NSF4SL | AdamsonGI | matched_in_sldb | -0.0214 ± 0.0433 (n=941) | -0.0214 ± 0.0663 (n=1,496) | -0.0401 ± 0.1153 (n=140) |
| NSF4SL | Corn | matched | +0.0135 ± 0.0183 (n=16,614) | -0.0066 ± 0.0292 (n=29,200) | -0.0390 ± 0.0385 (n=3,117) |
| NSF4SL | Corn | matched_incl_sldb | +0.0151 ± 0.0194 (n=16,864) | -0.0012 ± 0.0290 (n=29,624) | -0.0349 ± 0.0391 (n=3,160) |
| NSF4SL | Corn | matched_in_sldb | -0.0309 ± 0.0650 (n=250) | +0.0488 ± 0.0516 (n=424) | -0.0530 ± 0.0910 (n=43) |
| NSF4SL | Gilbert | matched | -0.0068 ± 0.0142 (n=1,383) | -0.0011 ± 0.0351 (n=2,590) | +0.0767 ± 0.0728 (n=299) |
| NSF4SL | Gilbert | matched_incl_sldb | +0.0387 ± 0.0186 (n=18,786) | +0.0258 ± 0.0239 (n=32,886) | +0.0361 ± 0.0364 (n=3,486) |
| NSF4SL | Gilbert | matched_in_sldb | +0.0418 ± 0.0206 (n=17,403) | +0.0283 ± 0.0248 (n=30,296) | +0.0334 ± 0.0337 (n=3,187) |
| DegreePrior | Adamson | matched | +0.2124 ± 0.0381 (n=14,727) | +0.2497 ± 0.0662 (n=25,782) | +0.2951 ± 0.1280 (n=2,709) |
| DegreePrior | Adamson | matched_incl_sldb | +0.2383 ± 0.0354 (n=15,669) | +0.2777 ± 0.0608 (n=27,317) | +0.3208 ± 0.1196 (n=2,850) |
| DegreePrior | Adamson | matched_in_sldb | +0.0632 ± 0.0737 (n=942) | +0.0259 ± 0.0604 (n=1,535) | +0.0248 ± 0.2070 (n=141) |
| DegreePrior | AdamsonGI | matched | +0.0174 ± 0.0135 (n=14,586) | +0.0054 ± 0.0099 (n=27,404) | -0.0073 ± 0.0271 (n=3,181) |
| DegreePrior | AdamsonGI | matched_incl_sldb | +0.0121 ± 0.0103 (n=15,527) | +0.0018 ± 0.0115 (n=28,900) | -0.0067 ± 0.0294 (n=3,321) |
| DegreePrior | AdamsonGI | matched_in_sldb | +0.0228 ± 0.0430 (n=941) | +0.0192 ± 0.0778 (n=1,496) | +0.0314 ± 0.0964 (n=140) |
| DegreePrior | Corn | matched | +0.0526 ± 0.0070 (n=16,614) | +0.0713 ± 0.0052 (n=29,200) | +0.0981 ± 0.0172 (n=3,117) |
| DegreePrior | Corn | matched_incl_sldb | +0.0548 ± 0.0080 (n=16,864) | +0.0752 ± 0.0052 (n=29,624) | +0.1020 ± 0.0176 (n=3,160) |
| DegreePrior | Corn | matched_in_sldb | -0.0091 ± 0.0721 (n=250) | +0.0114 ± 0.0307 (n=424) | -0.0080 ± 0.1178 (n=43) |
| DegreePrior | Gilbert | matched | -0.0261 ± 0.0148 (n=1,383) | +0.0078 ± 0.0279 (n=2,590) | +0.0480 ± 0.0634 (n=299) |
| DegreePrior | Gilbert | matched_incl_sldb | +0.0558 ± 0.0254 (n=18,786) | +0.0528 ± 0.0370 (n=32,886) | +0.0562 ± 0.0511 (n=3,486) |
| DegreePrior | Gilbert | matched_in_sldb | +0.0615 ± 0.0271 (n=17,403) | +0.0570 ± 0.0389 (n=30,296) | +0.0584 ± 0.0508 (n=3,187) |
| SynLethDB_label | Adamson | matched_in_sldb | -0.0804 ± 0.0000 (n=942) | -0.1157 ± 0.0000 (n=1,535) | -0.0658 ± 0.0000 (n=141) |
| SynLethDB_label | AdamsonGI | matched_in_sldb | +0.0400 ± 0.0000 (n=941) | +0.0504 ± 0.0000 (n=1,496) | +0.0530 ± 0.0000 (n=140) |
| SynLethDB_label | Corn | matched_in_sldb | +0.0769 ± 0.0000 (n=250) | -0.0041 ± 0.0000 (n=424) | +0.1372 ± 0.0000 (n=43) |
| SynLethDB_label | Gilbert | matched_in_sldb | +0.2065 ± 0.0000 (n=17,403) | +0.1984 ± 0.0000 (n=30,296) | +0.1722 ± 0.0000 (n=3,187) |

## SynLethDB label vs screen target

On the pairs SynLethDB labels: AUROC of the target (higher = more SL) for SynLethDB SL vs screened non-SL, and Spearman(label, target). CIs: bootstrap_ci.py --part external (ci_sldb_agreement.csv).

| Screen | screen_role | Scope | Split | n (SL / non-SL) | AUROC | Spearman |
|---|---|---|---|---|---|---|
| Adamson | external | all_pairs | any | 4,794 (612 / 4,182) | 0.4183 | -0.0944 |
| Adamson | external | all_pairs_genes_txt | any | 4,794 (612 / 4,182) | 0.4183 | -0.0944 |
| Adamson | external | test_genes_txt | cv1 | 942 (111 / 831) | 0.4280 | -0.0804 |
| Adamson | external | test_genes_txt | cv2 | 1,535 (194 / 1,341) | 0.3995 | -0.1157 |
| Adamson | external | test_genes_txt | cv3 | 141 (14 / 127) | 0.4364 | -0.0658 |
| AdamsonGI | external | all_pairs | any | 4,705 (612 / 4,093) | 0.5559 | +0.0652 |
| AdamsonGI | external | all_pairs_genes_txt | any | 4,705 (612 / 4,093) | 0.5559 | +0.0652 |
| AdamsonGI | external | test_genes_txt | cv1 | 941 (119 / 822) | 0.5347 | +0.0400 |
| AdamsonGI | external | test_genes_txt | cv2 | 1,496 (335 / 1,161) | 0.5349 | +0.0504 |
| AdamsonGI | external | test_genes_txt | cv3 | 140 (52 / 88) | 0.5317 | +0.0530 |
| Corn | external | all_pairs | any | 1,251 (240 / 1,011) | 0.5438 | +0.0623 |
| Corn | external | all_pairs_genes_txt | any | 1,251 (240 / 1,011) | 0.5438 | +0.0623 |
| Corn | external | test_genes_txt | cv1 | 250 (50 / 200) | 0.5524 | +0.0769 |
| Corn | external | test_genes_txt | cv2 | 424 (96 / 328) | 0.4973 | -0.0041 |
| Corn | external | test_genes_txt | cv3 | 43 (9 / 34) | 0.5948 | +0.1372 |
| Gilbert | in_distribution_check | all_pairs | any | 87,113 (6,021 / 81,092) | 0.7425 | +0.2131 |
| Gilbert | in_distribution_check | all_pairs_genes_txt | any | 87,113 (6,021 / 81,092) | 0.7425 | +0.2131 |
| Gilbert | in_distribution_check | test_genes_txt | cv1 | 17,403 (1,211 / 16,192) | 0.7342 | +0.2065 |
| Gilbert | in_distribution_check | test_genes_txt | cv2 | 30,296 (2,368 / 27,928) | 0.7133 | +0.1984 |
| Gilbert | in_distribution_check | test_genes_txt | cv3 | 3,187 (276 / 2,911) | 0.6768 | +0.1722 |

## Zero-shot, SLeuth on wider row sets (secondary)

`all_sldb_filtered` = test, not in SynLethDB (SL or screened non-SL; no genes.txt restriction): the row set of the paper's fine-tuning table. Baselines cannot score genes outside genes.txt, so these rows are SLeuth-only and NOT comparable across models: compare models on the matched rows above.

| Variant | Screen | Row set | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|---|
| siamese_sl_pd_pooled | Adamson | all_sldb_filtered | -0.0617 ± 0.0513 (n=28,375) | -0.0625 ± 0.0575 (n=51,107) | -0.0729 ± 0.0711 (n=5,637) |
| siamese_sl_pd_pooled | Adamson | all_unfiltered | -0.0713 ± 0.0522 (n=29,317) | -0.0721 ± 0.0584 (n=52,642) | -0.0814 ± 0.0720 (n=5,778) |
| siamese_sl_pd_pooled | AdamsonGI | all_sldb_filtered | +0.0482 ± 0.0172 (n=27,730) | +0.0437 ± 0.0224 (n=50,069) | +0.0444 ± 0.0310 (n=5,531) |
| siamese_sl_pd_pooled | AdamsonGI | all_unfiltered | +0.0501 ± 0.0163 (n=28,671) | +0.0457 ± 0.0225 (n=51,565) | +0.0468 ± 0.0314 (n=5,671) |
| siamese_sl_pd_pooled | Corn | all_sldb_filtered | +0.0356 ± 0.0093 (n=29,707) | +0.0415 ± 0.0131 (n=53,732) | +0.0600 ± 0.0230 (n=5,952) |
| siamese_sl_pd_pooled | Corn | all_unfiltered | +0.0356 ± 0.0093 (n=29,957) | +0.0415 ± 0.0130 (n=54,156) | +0.0603 ± 0.0225 (n=5,995) |
| siamese_sl_pd_pooled | Gilbert | all_sldb_filtered | +0.0546 ± 0.0353 (n=2,533) | +0.1141 ± 0.0295 (n=5,482) | +0.1753 ± 0.0360 (n=729) |
| siamese_sl_pd_pooled | Gilbert | all_unfiltered | +0.1268 ± 0.0217 (n=19,936) | +0.1284 ± 0.0328 (n=35,778) | +0.1450 ± 0.0457 (n=3,916) |
| siamese_sl | Adamson | all_sldb_filtered | -0.1987 ± 0.0351 (n=28,375) | -0.2135 ± 0.0348 (n=51,107) | -0.2431 ± 0.0436 (n=5,637) |
| siamese_sl | Adamson | all_unfiltered | -0.2052 ± 0.0369 (n=29,317) | -0.2179 ± 0.0366 (n=52,642) | -0.2454 ± 0.0470 (n=5,778) |
| siamese_sl | AdamsonGI | all_sldb_filtered | +0.0587 ± 0.0168 (n=27,730) | +0.0578 ± 0.0211 (n=50,069) | +0.0704 ± 0.0304 (n=5,531) |
| siamese_sl | AdamsonGI | all_unfiltered | +0.0619 ± 0.0168 (n=28,671) | +0.0618 ± 0.0220 (n=51,565) | +0.0773 ± 0.0319 (n=5,671) |
| siamese_sl | Corn | all_sldb_filtered | +0.0452 ± 0.0091 (n=29,707) | +0.0534 ± 0.0113 (n=53,732) | +0.0588 ± 0.0228 (n=5,952) |
| siamese_sl | Corn | all_unfiltered | +0.0459 ± 0.0092 (n=29,957) | +0.0543 ± 0.0113 (n=54,156) | +0.0594 ± 0.0221 (n=5,995) |
| siamese_sl | Gilbert | all_sldb_filtered | +0.0314 ± 0.0271 (n=2,533) | +0.0793 ± 0.0268 (n=5,482) | +0.1224 ± 0.0540 (n=729) |
| siamese_sl | Gilbert | all_unfiltered | +0.1146 ± 0.0133 (n=19,936) | +0.1144 ± 0.0175 (n=35,778) | +0.1098 ± 0.0270 (n=3,916) |

## Fine-tuning (SLeuth variants)

**LP_last and Full start from an OLS-calibrated head**: each fold's (a, c) is warm-started from the closed-form OLS fit of target on the raw logit over the screen's train split. `zero_shot_raw` is the uncalibrated logit (the zero-shot table above); `zero_shot_ols` is that calibrated starting point, which differs from raw only by the sign of a (a < 0 flips it). Judge what fine-tuning adds against `zero_shot_ols`.

**Scratch** is the from-scratch baseline: the same architecture, fold feature space, split and fine-tuning schedule as Full, freshly initialised (no SynLethDB weight), its (a, c) warm-started from OLS on its own initial logits. Full minus Scratch is what pretraining adds.

### siamese_sl_pd_pooled — matched (test, not in SynLethDB (SL or screened non-SL), both genes in genes.txt)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.0843 ± 0.0496 | -0.0789 ± 0.0648 | -0.0882 ± 0.0932 |
| Adamson | zero_shot_ols | +0.0843 ± 0.0496 | +0.0672 ± 0.0769 | +0.0655 ± 0.1104 |
| Adamson | LP_last | +0.7523 ± 0.0270 | +0.4564 ± 0.0494 | +0.3407 ± 0.0824 |
| Adamson | Full | +0.9875 ± 0.0003 | +0.6288 ± 0.0204 | +0.2015 ± 0.1826 |
| Adamson | Scratch | +0.9823 ± 0.0058 | +0.5939 ± 0.0206 | +0.2679 ± 0.0945 |
| AdamsonGI | zero_shot_raw | +0.0623 ± 0.0196 | +0.0661 ± 0.0298 | +0.0736 ± 0.0470 |
| AdamsonGI | zero_shot_ols | +0.0623 ± 0.0196 | +0.0661 ± 0.0298 | +0.0736 ± 0.0470 |
| AdamsonGI | LP_last | +0.2395 ± 0.0051 | +0.1251 ± 0.0355 | +0.0854 ± 0.0565 |
| AdamsonGI | Full | +0.4532 ± 0.0067 | +0.2195 ± 0.0194 | +0.0736 ± 0.0470 |
| AdamsonGI | Scratch | +0.4479 ± 0.0066 | +0.2231 ± 0.0072 | +0.0727 ± 0.0377 |
| Corn | zero_shot_raw | +0.0409 ± 0.0167 | +0.0471 ± 0.0196 | +0.0660 ± 0.0263 |
| Corn | zero_shot_ols | +0.0409 ± 0.0167 | +0.0363 ± 0.0358 | +0.0528 ± 0.0476 |
| Corn | LP_last | +0.2785 ± 0.0071 | +0.1217 ± 0.0170 | +0.0870 ± 0.0276 |
| Corn | Full | +0.4832 ± 0.0022 | +0.2246 ± 0.0165 | +0.0393 ± 0.0395 |
| Corn | Scratch | +0.4782 ± 0.0023 | +0.2359 ± 0.0075 | +0.0443 ± 0.0189 |
| Gilbert | zero_shot_raw | +0.0162 ± 0.0334 | +0.0379 ± 0.0429 | +0.1006 ± 0.0462 |
| Gilbert | zero_shot_ols | +0.0162 ± 0.0334 | +0.0379 ± 0.0429 | +0.1006 ± 0.0462 |
| Gilbert | LP_last | +0.2194 ± 0.0290 | +0.1270 ± 0.0610 | +0.0731 ± 0.0608 |
| Gilbert | Full | +0.4345 ± 0.0122 | +0.1983 ± 0.0261 | +0.1015 ± 0.0445 |
| Gilbert | Scratch | +0.4235 ± 0.0093 | +0.1983 ± 0.0465 | +0.0732 ± 0.0360 |

### siamese_sl_pd_pooled — all_sldb_filtered (test, not in SynLethDB (SL or screened non-SL; no genes.txt restriction))

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.0617 ± 0.0513 | -0.0625 ± 0.0575 | -0.0729 ± 0.0711 |
| Adamson | zero_shot_ols | +0.0617 ± 0.0513 | +0.0684 ± 0.0504 | +0.0784 ± 0.0651 |
| Adamson | LP_last | +0.7387 ± 0.0256 | +0.4404 ± 0.0404 | +0.3498 ± 0.0712 |
| Adamson | Full | +0.9873 ± 0.0002 | +0.6297 ± 0.0108 | +0.2071 ± 0.1579 |
| Adamson | Scratch | +0.9822 ± 0.0055 | +0.6000 ± 0.0088 | +0.2702 ± 0.1083 |
| AdamsonGI | zero_shot_raw | +0.0482 ± 0.0172 | +0.0437 ± 0.0224 | +0.0444 ± 0.0310 |
| AdamsonGI | zero_shot_ols | +0.0482 ± 0.0172 | +0.0437 ± 0.0224 | +0.0444 ± 0.0310 |
| AdamsonGI | LP_last | +0.2191 ± 0.0026 | +0.1128 ± 0.0381 | +0.0626 ± 0.0461 |
| AdamsonGI | Full | +0.4488 ± 0.0066 | +0.2005 ± 0.0174 | +0.0444 ± 0.0310 |
| AdamsonGI | Scratch | +0.4434 ± 0.0061 | +0.2023 ± 0.0073 | +0.0665 ± 0.0230 |
| Corn | zero_shot_raw | +0.0356 ± 0.0093 | +0.0415 ± 0.0131 | +0.0600 ± 0.0230 |
| Corn | zero_shot_ols | +0.0356 ± 0.0093 | +0.0332 ± 0.0281 | +0.0496 ± 0.0409 |
| Corn | LP_last | +0.2631 ± 0.0079 | +0.1171 ± 0.0147 | +0.0661 ± 0.0214 |
| Corn | Full | +0.4655 ± 0.0020 | +0.2166 ± 0.0143 | +0.0398 ± 0.0393 |
| Corn | Scratch | +0.4598 ± 0.0030 | +0.2247 ± 0.0032 | +0.0336 ± 0.0090 |
| Gilbert | zero_shot_raw | +0.0546 ± 0.0353 | +0.1141 ± 0.0295 | +0.1753 ± 0.0360 |
| Gilbert | zero_shot_ols | +0.0546 ± 0.0353 | +0.1141 ± 0.0295 | +0.1753 ± 0.0360 |
| Gilbert | LP_last | +0.3109 ± 0.0128 | +0.2055 ± 0.0402 | +0.1946 ± 0.0355 |
| Gilbert | Full | +0.5322 ± 0.0074 | +0.2710 ± 0.0221 | +0.1785 ± 0.0601 |
| Gilbert | Scratch | +0.5270 ± 0.0025 | +0.2382 ± 0.0473 | +0.1031 ± 0.0548 |

### siamese_sl_pd_pooled — all_unfiltered (all test rows)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.0713 ± 0.0522 | -0.0721 ± 0.0584 | -0.0814 ± 0.0720 |
| Adamson | zero_shot_ols | +0.0713 ± 0.0522 | +0.0754 ± 0.0540 | +0.0850 ± 0.0678 |
| Adamson | LP_last | +0.7451 ± 0.0253 | +0.4495 ± 0.0417 | +0.3595 ± 0.0709 |
| Adamson | Full | +0.9874 ± 0.0002 | +0.6349 ± 0.0110 | +0.2156 ± 0.1565 |
| Adamson | Scratch | +0.9821 ± 0.0058 | +0.6046 ± 0.0093 | +0.2744 ± 0.1093 |
| AdamsonGI | zero_shot_raw | +0.0501 ± 0.0163 | +0.0457 ± 0.0225 | +0.0468 ± 0.0314 |
| AdamsonGI | zero_shot_ols | +0.0501 ± 0.0163 | +0.0457 ± 0.0225 | +0.0468 ± 0.0314 |
| AdamsonGI | LP_last | +0.2256 ± 0.0028 | +0.1154 ± 0.0392 | +0.0644 ± 0.0462 |
| AdamsonGI | Full | +0.4549 ± 0.0064 | +0.2059 ± 0.0184 | +0.0468 ± 0.0314 |
| AdamsonGI | Scratch | +0.4498 ± 0.0058 | +0.2076 ± 0.0077 | +0.0665 ± 0.0254 |
| Corn | zero_shot_raw | +0.0356 ± 0.0093 | +0.0415 ± 0.0130 | +0.0603 ± 0.0225 |
| Corn | zero_shot_ols | +0.0356 ± 0.0093 | +0.0333 ± 0.0279 | +0.0497 ± 0.0409 |
| Corn | LP_last | +0.2644 ± 0.0082 | +0.1191 ± 0.0148 | +0.0678 ± 0.0219 |
| Corn | Full | +0.4665 ± 0.0020 | +0.2185 ± 0.0141 | +0.0397 ± 0.0394 |
| Corn | Scratch | +0.4608 ± 0.0029 | +0.2267 ± 0.0032 | +0.0327 ± 0.0085 |
| Gilbert | zero_shot_raw | +0.1268 ± 0.0217 | +0.1284 ± 0.0328 | +0.1450 ± 0.0457 |
| Gilbert | zero_shot_ols | +0.1268 ± 0.0217 | +0.1284 ± 0.0328 | +0.1450 ± 0.0457 |
| Gilbert | LP_last | +0.3145 ± 0.0174 | +0.2273 ± 0.0318 | +0.1988 ± 0.0393 |
| Gilbert | Full | +0.5320 ± 0.0048 | +0.3147 ± 0.0088 | +0.1789 ± 0.0706 |
| Gilbert | Scratch | +0.5184 ± 0.0044 | +0.2799 ± 0.0555 | +0.1622 ± 0.0310 |

### siamese_sl — matched (test, not in SynLethDB (SL or screened non-SL), both genes in genes.txt)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.2069 ± 0.0348 | -0.2106 ± 0.0313 | -0.2385 ± 0.0458 |
| Adamson | zero_shot_ols | +0.2069 ± 0.0348 | +0.2106 ± 0.0313 | +0.2385 ± 0.0458 |
| Adamson | LP_last | +0.7131 ± 0.0400 | +0.3827 ± 0.0664 | +0.2304 ± 0.0654 |
| Adamson | Full | +0.9865 ± 0.0006 | +0.6203 ± 0.0272 | +0.3834 ± 0.0483 |
| Adamson | Scratch | +0.9854 ± 0.0006 | +0.5927 ± 0.0157 | +0.2922 ± 0.0898 |
| AdamsonGI | zero_shot_raw | +0.0644 ± 0.0176 | +0.0668 ± 0.0239 | +0.0767 ± 0.0321 |
| AdamsonGI | zero_shot_ols | +0.0644 ± 0.0176 | +0.0668 ± 0.0239 | +0.0767 ± 0.0321 |
| AdamsonGI | LP_last | +0.2240 ± 0.0179 | +0.0998 ± 0.0309 | +0.0809 ± 0.0337 |
| AdamsonGI | Full | +0.4644 ± 0.0054 | +0.2252 ± 0.0145 | +0.0827 ± 0.0382 |
| AdamsonGI | Scratch | +0.4522 ± 0.0070 | +0.2177 ± 0.0151 | +0.0718 ± 0.0512 |
| Corn | zero_shot_raw | +0.0465 ± 0.0113 | +0.0521 ± 0.0111 | +0.0538 ± 0.0198 |
| Corn | zero_shot_ols | +0.0465 ± 0.0113 | +0.0521 ± 0.0111 | +0.0538 ± 0.0198 |
| Corn | LP_last | +0.2591 ± 0.0164 | +0.1036 ± 0.0259 | +0.0770 ± 0.0327 |
| Corn | Full | +0.4818 ± 0.0065 | +0.2330 ± 0.0121 | +0.0332 ± 0.0254 |
| Corn | Scratch | +0.4804 ± 0.0047 | +0.2534 ± 0.0068 | +0.0607 ± 0.0109 |
| Gilbert | zero_shot_raw | +0.0220 ± 0.0187 | +0.0313 ± 0.0343 | +0.0783 ± 0.0627 |
| Gilbert | zero_shot_ols | +0.0220 ± 0.0187 | +0.0313 ± 0.0343 | +0.0783 ± 0.0627 |
| Gilbert | LP_last | +0.1879 ± 0.0296 | +0.0881 ± 0.0369 | +0.0108 ± 0.0513 |
| Gilbert | Full | +0.4415 ± 0.0127 | +0.1923 ± 0.0233 | +0.0871 ± 0.0248 |
| Gilbert | Scratch | +0.4344 ± 0.0064 | +0.2231 ± 0.0092 | +0.0705 ± 0.0198 |

### siamese_sl — all_sldb_filtered (test, not in SynLethDB (SL or screened non-SL; no genes.txt restriction))

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.1987 ± 0.0351 | -0.2135 ± 0.0348 | -0.2431 ± 0.0436 |
| Adamson | zero_shot_ols | +0.1987 ± 0.0351 | +0.2135 ± 0.0348 | +0.2431 ± 0.0436 |
| Adamson | LP_last | +0.7101 ± 0.0427 | +0.3859 ± 0.0742 | +0.2387 ± 0.0514 |
| Adamson | Full | +0.9863 ± 0.0006 | +0.6265 ± 0.0134 | +0.3800 ± 0.0271 |
| Adamson | Scratch | +0.9852 ± 0.0005 | +0.6056 ± 0.0126 | +0.2924 ± 0.1075 |
| AdamsonGI | zero_shot_raw | +0.0587 ± 0.0168 | +0.0578 ± 0.0211 | +0.0704 ± 0.0304 |
| AdamsonGI | zero_shot_ols | +0.0587 ± 0.0168 | +0.0578 ± 0.0211 | +0.0704 ± 0.0304 |
| AdamsonGI | LP_last | +0.2032 ± 0.0170 | +0.0921 ± 0.0259 | +0.0746 ± 0.0306 |
| AdamsonGI | Full | +0.4594 ± 0.0045 | +0.2038 ± 0.0146 | +0.0702 ± 0.0302 |
| AdamsonGI | Scratch | +0.4486 ± 0.0077 | +0.1928 ± 0.0145 | +0.0627 ± 0.0303 |
| Corn | zero_shot_raw | +0.0452 ± 0.0091 | +0.0534 ± 0.0113 | +0.0588 ± 0.0228 |
| Corn | zero_shot_ols | +0.0452 ± 0.0091 | +0.0534 ± 0.0113 | +0.0588 ± 0.0228 |
| Corn | LP_last | +0.2459 ± 0.0176 | +0.0954 ± 0.0223 | +0.0578 ± 0.0218 |
| Corn | Full | +0.4639 ± 0.0067 | +0.2239 ± 0.0127 | +0.0342 ± 0.0220 |
| Corn | Scratch | +0.4610 ± 0.0044 | +0.2371 ± 0.0094 | +0.0460 ± 0.0153 |
| Gilbert | zero_shot_raw | +0.0314 ± 0.0271 | +0.0793 ± 0.0268 | +0.1224 ± 0.0540 |
| Gilbert | zero_shot_ols | +0.0314 ± 0.0271 | +0.0793 ± 0.0268 | +0.1224 ± 0.0540 |
| Gilbert | LP_last | +0.2503 ± 0.0471 | +0.1320 ± 0.0468 | +0.0962 ± 0.0695 |
| Gilbert | Full | +0.5369 ± 0.0078 | +0.2422 ± 0.0163 | +0.1565 ± 0.0402 |
| Gilbert | Scratch | +0.5308 ± 0.0030 | +0.2604 ± 0.0129 | +0.0940 ± 0.0349 |

### siamese_sl — all_unfiltered (all test rows)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.2052 ± 0.0369 | -0.2179 ± 0.0366 | -0.2454 ± 0.0470 |
| Adamson | zero_shot_ols | +0.2052 ± 0.0369 | +0.2179 ± 0.0366 | +0.2454 ± 0.0470 |
| Adamson | LP_last | +0.7178 ± 0.0418 | +0.3965 ± 0.0755 | +0.2454 ± 0.0524 |
| Adamson | Full | +0.9863 ± 0.0006 | +0.6310 ± 0.0137 | +0.3851 ± 0.0280 |
| Adamson | Scratch | +0.9852 ± 0.0005 | +0.6104 ± 0.0134 | +0.2980 ± 0.1100 |
| AdamsonGI | zero_shot_raw | +0.0619 ± 0.0168 | +0.0618 ± 0.0220 | +0.0773 ± 0.0319 |
| AdamsonGI | zero_shot_ols | +0.0619 ± 0.0168 | +0.0618 ± 0.0220 | +0.0773 ± 0.0319 |
| AdamsonGI | LP_last | +0.2078 ± 0.0167 | +0.0927 ± 0.0277 | +0.0789 ± 0.0326 |
| AdamsonGI | Full | +0.4653 ± 0.0042 | +0.2101 ± 0.0149 | +0.0764 ± 0.0309 |
| AdamsonGI | Scratch | +0.4547 ± 0.0075 | +0.1978 ± 0.0156 | +0.0624 ± 0.0317 |
| Corn | zero_shot_raw | +0.0459 ± 0.0092 | +0.0543 ± 0.0113 | +0.0594 ± 0.0221 |
| Corn | zero_shot_ols | +0.0459 ± 0.0092 | +0.0543 ± 0.0113 | +0.0594 ± 0.0221 |
| Corn | LP_last | +0.2473 ± 0.0173 | +0.0980 ± 0.0227 | +0.0598 ± 0.0226 |
| Corn | Full | +0.4648 ± 0.0067 | +0.2257 ± 0.0128 | +0.0336 ± 0.0228 |
| Corn | Scratch | +0.4619 ± 0.0045 | +0.2391 ± 0.0091 | +0.0457 ± 0.0156 |
| Gilbert | zero_shot_raw | +0.1146 ± 0.0133 | +0.1144 ± 0.0175 | +0.1098 ± 0.0270 |
| Gilbert | zero_shot_ols | +0.1146 ± 0.0133 | +0.1144 ± 0.0175 | +0.1098 ± 0.0270 |
| Gilbert | LP_last | +0.2802 ± 0.0451 | +0.1910 ± 0.0413 | +0.1446 ± 0.0566 |
| Gilbert | Full | +0.5362 ± 0.0029 | +0.3129 ± 0.0089 | +0.1866 ± 0.0498 |
| Gilbert | Scratch | +0.5199 ± 0.0020 | +0.3008 ± 0.0049 | +0.1427 ± 0.0577 |

## Per-fold zero-shot values (matched rows)

| Model | Screen | Split | n | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|---|---|---|
| siamese_sl_pd_pooled | Adamson | cv1 | 14,727 | -0.0894 | -0.1334 | -0.0262 | -0.0295 | -0.1432 |
| siamese_sl_pd_pooled | Adamson | cv2 | 25,782 | -0.0683 | -0.1378 | -0.0294 | +0.0069 | -0.1661 |
| siamese_sl_pd_pooled | Adamson | cv3 | 2,709 | -0.0690 | -0.1684 | -0.0568 | +0.0592 | -0.2061 |
| siamese_sl_pd_pooled | AdamsonGI | cv1 | 14,586 | +0.0784 | +0.0605 | +0.0251 | +0.0759 | +0.0717 |
| siamese_sl_pd_pooled | AdamsonGI | cv2 | 27,404 | +0.0869 | +0.0401 | +0.0227 | +0.1023 | +0.0786 |
| siamese_sl_pd_pooled | AdamsonGI | cv3 | 3,181 | +0.1102 | +0.0046 | +0.0293 | +0.1137 | +0.1104 |
| siamese_sl_pd_pooled | Corn | cv1 | 16,614 | +0.0221 | +0.0682 | +0.0348 | +0.0510 | +0.0284 |
| siamese_sl_pd_pooled | Corn | cv2 | 29,200 | +0.0269 | +0.0665 | +0.0362 | +0.0747 | +0.0310 |
| siamese_sl_pd_pooled | Corn | cv3 | 3,117 | +0.0332 | +0.0861 | +0.0516 | +0.1061 | +0.0531 |
| siamese_sl_pd_pooled | Gilbert | cv1 | 1,383 | +0.0583 | -0.0239 | -0.0180 | +0.0166 | +0.0481 |
| siamese_sl_pd_pooled | Gilbert | cv2 | 2,590 | +0.1020 | +0.0353 | -0.0323 | +0.0494 | +0.0352 |
| siamese_sl_pd_pooled | Gilbert | cv3 | 299 | +0.1664 | +0.1013 | +0.0224 | +0.1150 | +0.0978 |
| siamese_sl | Adamson | cv1 | 14,727 | -0.2188 | -0.2010 | -0.2074 | -0.1495 | -0.2580 |
| siamese_sl | Adamson | cv2 | 25,782 | -0.2585 | -0.2010 | -0.1959 | -0.1670 | -0.2307 |
| siamese_sl | Adamson | cv3 | 2,709 | -0.3243 | -0.2479 | -0.2117 | -0.2030 | -0.2057 |
| siamese_sl | AdamsonGI | cv1 | 14,586 | +0.0677 | +0.0768 | +0.0346 | +0.0857 | +0.0570 |
| siamese_sl | AdamsonGI | cv2 | 27,404 | +0.0691 | +0.0706 | +0.0216 | +0.0900 | +0.0827 |
| siamese_sl | AdamsonGI | cv3 | 3,181 | +0.0773 | +0.0459 | +0.0374 | +0.1009 | +0.1221 |
| siamese_sl | Corn | cv1 | 16,614 | +0.0582 | +0.0622 | +0.0363 | +0.0361 | +0.0396 |
| siamese_sl | Corn | cv2 | 29,200 | +0.0614 | +0.0636 | +0.0349 | +0.0436 | +0.0571 |
| siamese_sl | Corn | cv3 | 3,117 | +0.0646 | +0.0772 | +0.0238 | +0.0378 | +0.0655 |
| siamese_sl | Gilbert | cv1 | 1,383 | +0.0499 | +0.0063 | -0.0032 | +0.0274 | +0.0294 |
| siamese_sl | Gilbert | cv2 | 2,590 | +0.0798 | +0.0221 | -0.0255 | +0.0346 | +0.0458 |
| siamese_sl | Gilbert | cv3 | 299 | +0.1381 | +0.0775 | -0.0409 | +0.1021 | +0.1147 |
| SLMGAE | Adamson | cv1 | 14,727 | -0.0695 | -0.0192 | +0.0177 | -0.0324 | -0.0515 |
| SLMGAE | Adamson | cv2 | 25,782 | -0.0876 | -0.0121 | +0.0095 | -0.0049 | -0.0638 |
| SLMGAE | Adamson | cv3 | 2,709 | -0.1086 | -0.0079 | -0.0008 | +0.0083 | -0.0829 |
| SLMGAE | AdamsonGI | cv1 | 14,586 | +0.0115 | +0.0129 | +0.0216 | +0.0294 | +0.0342 |
| SLMGAE | AdamsonGI | cv2 | 27,404 | +0.0269 | +0.0230 | +0.0464 | +0.0536 | +0.0591 |
| SLMGAE | AdamsonGI | cv3 | 3,181 | +0.0586 | +0.0337 | +0.0918 | +0.0981 | +0.1011 |
| SLMGAE | Corn | cv1 | 16,614 | +0.0506 | +0.0409 | +0.0539 | +0.0132 | +0.0586 |
| SLMGAE | Corn | cv2 | 29,200 | +0.0638 | +0.0495 | +0.0549 | +0.0128 | +0.0764 |
| SLMGAE | Corn | cv3 | 3,117 | +0.0823 | +0.0910 | +0.0699 | +0.0429 | +0.1072 |
| SLMGAE | Gilbert | cv1 | 1,383 | -0.0312 | -0.0159 | -0.0216 | +0.0007 | -0.0510 |
| SLMGAE | Gilbert | cv2 | 2,590 | +0.0009 | -0.0206 | -0.0632 | +0.0372 | -0.0248 |
| SLMGAE | Gilbert | cv3 | 299 | +0.0637 | -0.0281 | -0.0124 | +0.1014 | +0.0112 |
| SLGNN | Adamson | cv1 | 14,727 | -0.1273 | -0.1225 | -0.1099 | -0.0761 | -0.1356 |
| SLGNN | Adamson | cv2 | 25,782 | -0.1556 | -0.1703 | -0.1072 | -0.1013 | -0.1745 |
| SLGNN | Adamson | cv3 | 2,709 | -0.2253 | -0.2503 | -0.1223 | -0.1461 | -0.2512 |
| SLGNN | AdamsonGI | cv1 | 14,586 | +0.0671 | +0.0995 | +0.0601 | +0.0489 | +0.0781 |
| SLGNN | AdamsonGI | cv2 | 27,404 | +0.0486 | +0.0764 | +0.0475 | +0.0586 | +0.0664 |
| SLGNN | AdamsonGI | cv3 | 3,181 | +0.0362 | +0.0735 | +0.0557 | +0.0781 | +0.0747 |
| SLGNN | Corn | cv1 | 16,614 | +0.0516 | +0.0370 | +0.0158 | +0.0308 | +0.0372 |
| SLGNN | Corn | cv2 | 29,200 | +0.0378 | +0.0267 | +0.0080 | -0.0003 | +0.0150 |
| SLGNN | Corn | cv3 | 3,117 | +0.0281 | +0.0502 | +0.0005 | -0.0214 | +0.0213 |
| SLGNN | Gilbert | cv1 | 1,383 | -0.0051 | -0.0362 | -0.0288 | -0.0131 | -0.0484 |
| SLGNN | Gilbert | cv2 | 2,590 | +0.0274 | -0.0020 | -0.0255 | +0.0198 | +0.0074 |
| SLGNN | Gilbert | cv3 | 299 | +0.0785 | +0.0226 | -0.0230 | +0.0906 | +0.0975 |
| NSF4SL | Adamson | cv1 | 14,727 | -0.1674 | +0.1110 | +0.1861 | -0.0086 | +0.1979 |
| NSF4SL | Adamson | cv2 | 25,782 | -0.0940 | +0.1590 | +0.1852 | +0.0505 | +0.2144 |
| NSF4SL | Adamson | cv3 | 2,709 | -0.0150 | +0.1431 | +0.1855 | +0.1050 | +0.2145 |
| NSF4SL | AdamsonGI | cv1 | 14,586 | +0.0118 | +0.0336 | +0.0009 | -0.0189 | +0.0087 |
| NSF4SL | AdamsonGI | cv2 | 27,404 | +0.0173 | +0.0253 | +0.0034 | -0.0221 | +0.0047 |
| NSF4SL | AdamsonGI | cv3 | 3,181 | +0.0273 | +0.0497 | +0.0192 | -0.0240 | +0.0204 |
| NSF4SL | Corn | cv1 | 16,614 | -0.0197 | +0.0272 | +0.0331 | +0.0120 | +0.0149 |
| NSF4SL | Corn | cv2 | 29,200 | -0.0380 | +0.0262 | +0.0202 | -0.0439 | +0.0023 |
| NSF4SL | Corn | cv3 | 3,117 | -0.0739 | -0.0143 | +0.0001 | -0.0962 | -0.0109 |
| NSF4SL | Gilbert | cv1 | 1,383 | +0.0089 | +0.0045 | -0.0211 | -0.0265 | -0.0001 |
| NSF4SL | Gilbert | cv2 | 2,590 | +0.0295 | +0.0171 | -0.0694 | +0.0140 | +0.0033 |
| NSF4SL | Gilbert | cv3 | 299 | +0.0804 | +0.0733 | -0.0523 | +0.1691 | +0.1129 |
| DegreePrior | Adamson | cv1 | 14,727 | +0.1984 | +0.2441 | +0.1841 | +0.1665 | +0.2687 |
| DegreePrior | Adamson | cv2 | 25,782 | +0.2383 | +0.2907 | +0.1297 | +0.2666 | +0.3231 |
| DegreePrior | Adamson | cv3 | 2,709 | +0.2922 | +0.3505 | +0.0498 | +0.3960 | +0.3868 |
| DegreePrior | AdamsonGI | cv1 | 14,586 | +0.0172 | +0.0099 | +0.0414 | +0.0177 | +0.0008 |
| DegreePrior | AdamsonGI | cv2 | 27,404 | -0.0037 | -0.0003 | +0.0020 | +0.0244 | +0.0047 |
| DegreePrior | AdamsonGI | cv3 | 3,181 | -0.0535 | -0.0105 | -0.0115 | +0.0129 | +0.0258 |
| DegreePrior | Corn | cv1 | 16,614 | +0.0432 | +0.0579 | +0.0622 | +0.0468 | +0.0531 |
| DegreePrior | Corn | cv2 | 29,200 | +0.0744 | +0.0723 | +0.0754 | +0.0732 | +0.0611 |
| DegreePrior | Corn | cv3 | 3,117 | +0.1226 | +0.1029 | +0.1027 | +0.0927 | +0.0697 |
| DegreePrior | Gilbert | cv1 | 1,383 | -0.0230 | -0.0365 | -0.0328 | +0.0014 | -0.0396 |
| DegreePrior | Gilbert | cv2 | 2,590 | +0.0101 | +0.0147 | -0.0456 | +0.0321 | +0.0278 |
| DegreePrior | Gilbert | cv3 | 299 | +0.0655 | -0.0146 | -0.0345 | +0.1335 | +0.0900 |

## Row sets

| screen | split | n_screen_pairs | n_test | n_test_not_sldb | n_matched | matched_digest | defined_by |
|---|---|---|---|---|---|---|---|
| Adamson | cv1 | 146583 | 29317 | 28375 | 14727 | 060b7b8f6b14ff90 | siamese_sl_pd_pooled |
| Adamson | cv2 | 146583 | 52642 | 51107 | 25782 | 2f088a3e0181304f | siamese_sl_pd_pooled |
| Adamson | cv3 | 146583 | 5778 | 5637 | 2709 | 1ecb5387448e5254 | siamese_sl_pd_pooled |
| AdamsonGI | cv1 | 143354 | 28671 | 27730 | 14586 | db1ef4f33d4e988c | siamese_sl_pd_pooled |
| AdamsonGI | cv2 | 143354 | 51565 | 50069 | 27404 | a9dbdf974fc83fcd | siamese_sl_pd_pooled |
| AdamsonGI | cv3 | 143354 | 5671 | 5531 | 3181 | b5b15fb532fc4601 | siamese_sl_pd_pooled |
| Corn | cv1 | 149787 | 29957 | 29707 | 16614 | 892bc59677bfaa5a | siamese_sl_pd_pooled |
| Corn | cv2 | 149787 | 54156 | 53732 | 29200 | 71bbcd79e90c1d83 | siamese_sl_pd_pooled |
| Corn | cv3 | 149787 | 5995 | 5952 | 3117 | 5de076347d2b8b5f | siamese_sl_pd_pooled |
| Gilbert | cv1 | 99681 | 19936 | 2533 | 1383 | 0471f3fabe1ea480 | siamese_sl_pd_pooled |
| Gilbert | cv2 | 99681 | 35778 | 5482 | 2590 | 2dfb388569612472 | siamese_sl_pd_pooled |
| Gilbert | cv3 | 99681 | 3916 | 729 | 299 | 77ec4d6e8d017d3b | siamese_sl_pd_pooled |

## Provenance

- **genes.txt**: <repo>/sl_comparison/folds/genes.txt (8,844 genes)
- **model CV (baselines / DegreePrior folds)**: cv3
- **SLeuth variants**: siamese_sl_pd_pooled = siamese_sl/results/var_pd_pooled__best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3; siamese_sl = siamese_sl/results/best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3
- **SLeuth scoring head (eval_finetuning config)**: siamese_sl_pd_pooled: effective_head={'Adamson': 'ANY (pooled)', 'AdamsonGI': 'ANY (pooled)', 'Corn': 'ANY (pooled)', 'Gilbert': 'ANY (pooled)'} (per_head=True, model_num_heads=1); siamese_sl: effective_head={'Adamson': 'K562', 'AdamsonGI': 'K562', 'Corn': 'OTHER', 'Gilbert': 'K562'} (per_head=True, model_num_heads=8)
- **baselines**: SLMGAE = SLMGAE-in-pytorch/results/slmgae_shared_cv3; SLGNN = SLGNN/results/slgnn_cv3; NSF4SL = NSF4SL/results/nsf4sl_cv3 (score = (M[i,j] + M[j,i]) / 2)
- **folds**: 0, 1, 2, 3, 4 (identical for every model)
- **DegreePrior**: log1p(deg_i) + log1p(deg_j) over <repo>/sl_comparison/folds/cv3/fold_k/train_pos.npy (float32, as degree_baseline.py)
- **sd**: population (ddof=0); eval_finetuning's own CSVs use ddof=1
- **eval fingerprint (checkpoint_sha256, fold_ids, 2 overlap sources)**: verified for every SLeuth variant and split
- **SynLethDB positives (overlap / agreement)**: <repo>/data/SL_SynLethDB_experimental.txt
- **SynLethDB screened negatives (overlap / agreement)**: <repo>/data/SL_SynLethDB_experimental_negatives.txt
- **fold training pairs (overlap)**: <repo>/sl_comparison/folds/cv3/fold_k/train_pos.npy + train_neg.npy
- **screen_role**: in_distribution_check when >= 0.5 of the screen's scored pairs are in SynLethDB, else external: Adamson=external, AdamsonGI=external, Corn=external, Gilbert=in_distribution_check
- **PCA arm of the SLeuth combos**: none (no PCA)

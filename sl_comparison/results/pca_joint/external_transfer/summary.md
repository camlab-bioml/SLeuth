# External-screen transfer (zero-shot + fine-tuning)

PCA arm: joint (PCA on the concatenation of all modalities (variance 0.8))

Metric: Spearman(score, target), target = -1 x screen phenotype (higher = more SL). Cells are mean ± population sd (ddof=0) over the training folds; per-fold values are in the CSVs and in the per-fold section below.

Screen splits: cv1 = random pairs, cv2 = >=1 gene unseen, cv3 = both genes unseen.

**The random / >=1-unseen / both-unseen columns only bear on fine-tuning.** They say how the SCREEN's pairs were divided into train / val / test (eval_finetuning --cv_type), i.e. which test genes the fine-tuned model saw in the screen's train split. A zero-shot score never uses the screen's train rows, so across the zero-shot columns only the test-row subset changes, not what the model was trained on.

Matched rows = test, not in SynLethDB (SL or screened non-SL), both genes in genes.txt. Every model below is scored on exactly these rows. SLeuth scores are RAW (uncalibrated) per-fold logits.

## Zero-shot, matched rows

### Adamson

| Model | random pairs (n=14,727) | >=1 gene unseen (n=25,782) | both genes unseen (n=2,709) |
|---|---|---|---|
| siamese_sl_pd_pooled | -0.0999 ± 0.0282 | -0.1046 ± 0.0504 | -0.1260 ± 0.0937 |
| siamese_sl | -0.1931 ± 0.0341 | -0.1935 ± 0.0638 | -0.1968 ± 0.1054 |
| SLMGAE | -0.0310 ± 0.0297 | -0.0318 ± 0.0373 | -0.0384 ± 0.0478 |
| SLGNN | -0.1143 ± 0.0208 | -0.1418 ± 0.0313 | -0.1990 ± 0.0543 |
| NSF4SL | +0.0638 ± 0.1370 | +0.1030 ± 0.1130 | +0.1266 ± 0.0800 |
| DegreePrior | +0.2124 ± 0.0381 | +0.2497 ± 0.0662 | +0.2951 ± 0.1280 |

### AdamsonGI

| Model | random pairs (n=14,586) | >=1 gene unseen (n=27,404) | both genes unseen (n=3,181) |
|---|---|---|---|
| siamese_sl_pd_pooled | +0.0676 ± 0.0072 | +0.0601 ± 0.0222 | +0.0503 ± 0.0333 |
| siamese_sl | +0.0526 ± 0.0140 | +0.0431 ± 0.0259 | +0.0341 ± 0.0357 |
| SLMGAE | +0.0219 ± 0.0089 | +0.0418 ± 0.0144 | +0.0767 ± 0.0263 |
| SLGNN | +0.0707 ± 0.0172 | +0.0595 ± 0.0109 | +0.0636 ± 0.0158 |
| NSF4SL | +0.0072 ± 0.0170 | +0.0057 ± 0.0161 | +0.0185 ± 0.0239 |
| DegreePrior | +0.0174 ± 0.0135 | +0.0054 ± 0.0099 | -0.0073 ± 0.0271 |

### Corn

| Model | random pairs (n=16,614) | >=1 gene unseen (n=29,200) | both genes unseen (n=3,117) |
|---|---|---|---|
| siamese_sl_pd_pooled | +0.0427 ± 0.0126 | +0.0417 ± 0.0223 | +0.0506 ± 0.0427 |
| siamese_sl | +0.0212 ± 0.0109 | +0.0291 ± 0.0081 | +0.0319 ± 0.0247 |
| SLMGAE | +0.0434 ± 0.0162 | +0.0515 ± 0.0214 | +0.0787 ± 0.0216 |
| SLGNN | +0.0345 ± 0.0116 | +0.0175 ± 0.0135 | +0.0157 ± 0.0245 |
| NSF4SL | +0.0135 ± 0.0183 | -0.0066 ± 0.0292 | -0.0390 ± 0.0385 |
| DegreePrior | +0.0526 ± 0.0070 | +0.0713 ± 0.0052 | +0.0981 ± 0.0172 |

### Gilbert

| Model | random pairs (n=1,383) | >=1 gene unseen (n=2,590) | both genes unseen (n=299) |
|---|---|---|---|
| siamese_sl_pd_pooled | +0.0162 ± 0.0218 | +0.0452 ± 0.0313 | +0.0752 ± 0.0699 |
| siamese_sl | +0.0131 ± 0.0322 | +0.0426 ± 0.0226 | +0.0925 ± 0.0730 |
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
| siamese_sl_pd_pooled | Adamson | matched | -0.0999 ± 0.0282 (n=14,727) | -0.1046 ± 0.0504 (n=25,782) | -0.1260 ± 0.0937 (n=2,709) |
| siamese_sl_pd_pooled | Adamson | matched_incl_sldb | -0.1190 ± 0.0272 (n=15,669) | -0.1233 ± 0.0507 (n=27,317) | -0.1424 ± 0.0940 (n=2,850) |
| siamese_sl_pd_pooled | Adamson | matched_in_sldb | -0.1790 ± 0.0719 (n=942) | -0.0989 ± 0.0969 (n=1,535) | -0.0148 ± 0.2246 (n=141) |
| siamese_sl_pd_pooled | AdamsonGI | matched | +0.0676 ± 0.0072 (n=14,586) | +0.0601 ± 0.0222 (n=27,404) | +0.0503 ± 0.0333 (n=3,181) |
| siamese_sl_pd_pooled | AdamsonGI | matched_incl_sldb | +0.0697 ± 0.0055 (n=15,527) | +0.0631 ± 0.0237 (n=28,900) | +0.0545 ± 0.0359 (n=3,321) |
| siamese_sl_pd_pooled | AdamsonGI | matched_in_sldb | +0.0836 ± 0.0474 (n=941) | +0.0978 ± 0.0628 (n=1,496) | +0.1067 ± 0.0877 (n=140) |
| siamese_sl_pd_pooled | Corn | matched | +0.0427 ± 0.0126 (n=16,614) | +0.0417 ± 0.0223 (n=29,200) | +0.0506 ± 0.0427 (n=3,117) |
| siamese_sl_pd_pooled | Corn | matched_incl_sldb | +0.0419 ± 0.0130 (n=16,864) | +0.0407 ± 0.0223 (n=29,624) | +0.0504 ± 0.0422 (n=3,160) |
| siamese_sl_pd_pooled | Corn | matched_in_sldb | +0.0017 ± 0.0684 (n=250) | -0.0178 ± 0.0448 (n=424) | -0.0678 ± 0.0676 (n=43) |
| siamese_sl_pd_pooled | Gilbert | matched | +0.0162 ± 0.0218 (n=1,383) | +0.0452 ± 0.0313 (n=2,590) | +0.0752 ± 0.0699 (n=299) |
| siamese_sl_pd_pooled | Gilbert | matched_incl_sldb | +0.0934 ± 0.0118 (n=18,786) | +0.0957 ± 0.0052 (n=32,886) | +0.0992 ± 0.0089 (n=3,486) |
| siamese_sl_pd_pooled | Gilbert | matched_in_sldb | +0.1046 ± 0.0134 (n=17,403) | +0.1009 ± 0.0074 (n=30,296) | +0.0980 ± 0.0121 (n=3,187) |
| siamese_sl | Adamson | matched | -0.1931 ± 0.0341 (n=14,727) | -0.1935 ± 0.0638 (n=25,782) | -0.1968 ± 0.1054 (n=2,709) |
| siamese_sl | Adamson | matched_incl_sldb | -0.2068 ± 0.0372 (n=15,669) | -0.2029 ± 0.0632 (n=27,317) | -0.2038 ± 0.1068 (n=2,850) |
| siamese_sl | Adamson | matched_in_sldb | -0.1516 ± 0.0673 (n=942) | -0.0505 ± 0.0673 (n=1,535) | +0.0534 ± 0.1040 (n=141) |
| siamese_sl | AdamsonGI | matched | +0.0526 ± 0.0140 (n=14,586) | +0.0431 ± 0.0259 (n=27,404) | +0.0341 ± 0.0357 (n=3,181) |
| siamese_sl | AdamsonGI | matched_incl_sldb | +0.0570 ± 0.0155 (n=15,527) | +0.0502 ± 0.0270 (n=28,900) | +0.0455 ± 0.0395 (n=3,321) |
| siamese_sl | AdamsonGI | matched_in_sldb | +0.0941 ± 0.0443 (n=941) | +0.1363 ± 0.0545 (n=1,496) | +0.2295 ± 0.1083 (n=140) |
| siamese_sl | Corn | matched | +0.0212 ± 0.0109 (n=16,614) | +0.0291 ± 0.0081 (n=29,200) | +0.0319 ± 0.0247 (n=3,117) |
| siamese_sl | Corn | matched_incl_sldb | +0.0225 ± 0.0113 (n=16,864) | +0.0304 ± 0.0083 (n=29,624) | +0.0319 ± 0.0227 (n=3,160) |
| siamese_sl | Corn | matched_in_sldb | -0.0055 ± 0.0885 (n=250) | -0.0545 ± 0.0987 (n=424) | -0.1906 ± 0.1516 (n=43) |
| siamese_sl | Gilbert | matched | +0.0131 ± 0.0322 (n=1,383) | +0.0426 ± 0.0226 (n=2,590) | +0.0925 ± 0.0730 (n=299) |
| siamese_sl | Gilbert | matched_incl_sldb | +0.1063 ± 0.0049 (n=18,786) | +0.1076 ± 0.0098 (n=32,886) | +0.1197 ± 0.0305 (n=3,486) |
| siamese_sl | Gilbert | matched_in_sldb | +0.1195 ± 0.0074 (n=17,403) | +0.1142 ± 0.0116 (n=30,296) | +0.1200 ± 0.0339 (n=3,187) |
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
| siamese_sl_pd_pooled | Adamson | all_sldb_filtered | -0.0823 ± 0.0233 (n=28,375) | -0.0899 ± 0.0496 (n=51,107) | -0.1080 ± 0.0939 (n=5,637) |
| siamese_sl_pd_pooled | Adamson | all_unfiltered | -0.0919 ± 0.0230 (n=29,317) | -0.0993 ± 0.0501 (n=52,642) | -0.1168 ± 0.0955 (n=5,778) |
| siamese_sl_pd_pooled | AdamsonGI | all_sldb_filtered | +0.0504 ± 0.0063 (n=27,730) | +0.0421 ± 0.0184 (n=50,069) | +0.0444 ± 0.0304 (n=5,531) |
| siamese_sl_pd_pooled | AdamsonGI | all_unfiltered | +0.0522 ± 0.0058 (n=28,671) | +0.0443 ± 0.0198 (n=51,565) | +0.0472 ± 0.0325 (n=5,671) |
| siamese_sl_pd_pooled | Corn | all_sldb_filtered | +0.0387 ± 0.0089 (n=29,707) | +0.0441 ± 0.0198 (n=53,732) | +0.0585 ± 0.0347 (n=5,952) |
| siamese_sl_pd_pooled | Corn | all_unfiltered | +0.0384 ± 0.0092 (n=29,957) | +0.0438 ± 0.0198 (n=54,156) | +0.0588 ± 0.0346 (n=5,995) |
| siamese_sl_pd_pooled | Gilbert | all_sldb_filtered | +0.0363 ± 0.0226 (n=2,533) | +0.1021 ± 0.0194 (n=5,482) | +0.1700 ± 0.0343 (n=729) |
| siamese_sl_pd_pooled | Gilbert | all_unfiltered | +0.0947 ± 0.0118 (n=19,936) | +0.1001 ± 0.0059 (n=35,778) | +0.1139 ± 0.0063 (n=3,916) |
| siamese_sl | Adamson | all_sldb_filtered | -0.1838 ± 0.0377 (n=28,375) | -0.1926 ± 0.0557 (n=51,107) | -0.2044 ± 0.0962 (n=5,637) |
| siamese_sl | Adamson | all_unfiltered | -0.1905 ± 0.0392 (n=29,317) | -0.1970 ± 0.0552 (n=52,642) | -0.2079 ± 0.0970 (n=5,778) |
| siamese_sl | AdamsonGI | all_sldb_filtered | +0.0424 ± 0.0097 (n=27,730) | +0.0332 ± 0.0213 (n=50,069) | +0.0280 ± 0.0365 (n=5,531) |
| siamese_sl | AdamsonGI | all_unfiltered | +0.0451 ± 0.0104 (n=28,671) | +0.0375 ± 0.0221 (n=51,565) | +0.0348 ± 0.0380 (n=5,671) |
| siamese_sl | Corn | all_sldb_filtered | +0.0242 ± 0.0151 (n=29,707) | +0.0320 ± 0.0100 (n=53,732) | +0.0387 ± 0.0236 (n=5,952) |
| siamese_sl | Corn | all_unfiltered | +0.0251 ± 0.0153 (n=29,957) | +0.0330 ± 0.0101 (n=54,156) | +0.0390 ± 0.0228 (n=5,995) |
| siamese_sl | Gilbert | all_sldb_filtered | +0.0421 ± 0.0345 (n=2,533) | +0.1038 ± 0.0240 (n=5,482) | +0.1471 ± 0.0248 (n=729) |
| siamese_sl | Gilbert | all_unfiltered | +0.1086 ± 0.0057 (n=19,936) | +0.1118 ± 0.0098 (n=35,778) | +0.1269 ± 0.0278 (n=3,916) |

## Fine-tuning (SLeuth variants)

**LP_last and Full start from an OLS-calibrated head**: each fold's (a, c) is warm-started from the closed-form OLS fit of target on the raw logit over the screen's train split. `zero_shot_raw` is the uncalibrated logit (the zero-shot table above); `zero_shot_ols` is that calibrated starting point, which differs from raw only by the sign of a (a < 0 flips it). Judge what fine-tuning adds against `zero_shot_ols`.

**Scratch** is the from-scratch baseline: the same architecture, fold feature space, split and fine-tuning schedule as Full, freshly initialised (no SynLethDB weight), its (a, c) warm-started from OLS on its own initial logits. Full minus Scratch is what pretraining adds.

### siamese_sl_pd_pooled — matched (test, not in SynLethDB (SL or screened non-SL), both genes in genes.txt)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.0999 ± 0.0282 | -0.1046 ± 0.0504 | -0.1260 ± 0.0937 |
| Adamson | zero_shot_ols | +0.0999 ± 0.0282 | +0.1046 ± 0.0504 | +0.1260 ± 0.0937 |
| Adamson | LP_last | +0.6877 ± 0.0128 | +0.3608 ± 0.0455 | +0.2236 ± 0.1041 |
| Adamson | Full | +0.9875 ± 0.0004 | +0.5923 ± 0.0315 | +0.1750 ± 0.1355 |
| Adamson | Scratch | +0.9867 ± 0.0002 | +0.5375 ± 0.0275 | +0.0285 ± 0.1620 |
| AdamsonGI | zero_shot_raw | +0.0676 ± 0.0072 | +0.0601 ± 0.0222 | +0.0503 ± 0.0333 |
| AdamsonGI | zero_shot_ols | +0.0676 ± 0.0072 | +0.0601 ± 0.0222 | +0.0503 ± 0.0333 |
| AdamsonGI | LP_last | +0.2304 ± 0.0089 | +0.1143 ± 0.0168 | +0.0757 ± 0.0340 |
| AdamsonGI | Full | +0.4504 ± 0.0060 | +0.2099 ± 0.0155 | +0.0503 ± 0.0333 |
| AdamsonGI | Scratch | +0.4472 ± 0.0066 | +0.2076 ± 0.0396 | +0.0577 ± 0.0426 |
| Corn | zero_shot_raw | +0.0427 ± 0.0126 | +0.0417 ± 0.0223 | +0.0506 ± 0.0427 |
| Corn | zero_shot_ols | +0.0427 ± 0.0126 | +0.0332 ± 0.0337 | +0.0453 ± 0.0483 |
| Corn | LP_last | +0.2625 ± 0.0176 | +0.0759 ± 0.0242 | +0.0290 ± 0.0317 |
| Corn | Full | +0.4822 ± 0.0038 | +0.2225 ± 0.0167 | +0.0368 ± 0.0486 |
| Corn | Scratch | +0.4774 ± 0.0033 | +0.2319 ± 0.0085 | +0.0431 ± 0.0239 |
| Gilbert | zero_shot_raw | +0.0162 ± 0.0218 | +0.0452 ± 0.0313 | +0.0752 ± 0.0699 |
| Gilbert | zero_shot_ols | +0.0162 ± 0.0218 | +0.0452 ± 0.0313 | +0.0752 ± 0.0699 |
| Gilbert | LP_last | +0.1844 ± 0.0260 | +0.1048 ± 0.0213 | +0.0379 ± 0.0689 |
| Gilbert | Full | +0.4371 ± 0.0100 | +0.1728 ± 0.0068 | +0.0736 ± 0.0708 |
| Gilbert | Scratch | +0.4239 ± 0.0110 | +0.1632 ± 0.0496 | +0.0103 ± 0.0546 |

### siamese_sl_pd_pooled — all_sldb_filtered (test, not in SynLethDB (SL or screened non-SL; no genes.txt restriction))

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.0823 ± 0.0233 | -0.0899 ± 0.0496 | -0.1080 ± 0.0939 |
| Adamson | zero_shot_ols | +0.0823 ± 0.0233 | +0.0899 ± 0.0496 | +0.1080 ± 0.0939 |
| Adamson | LP_last | +0.6834 ± 0.0115 | +0.3632 ± 0.0410 | +0.2284 ± 0.0916 |
| Adamson | Full | +0.9872 ± 0.0004 | +0.6157 ± 0.0215 | +0.1625 ± 0.1427 |
| Adamson | Scratch | +0.9864 ± 0.0002 | +0.5655 ± 0.0124 | +0.0243 ± 0.1591 |
| AdamsonGI | zero_shot_raw | +0.0504 ± 0.0063 | +0.0421 ± 0.0184 | +0.0444 ± 0.0304 |
| AdamsonGI | zero_shot_ols | +0.0504 ± 0.0063 | +0.0421 ± 0.0184 | +0.0444 ± 0.0304 |
| AdamsonGI | LP_last | +0.2100 ± 0.0087 | +0.1012 ± 0.0234 | +0.0705 ± 0.0285 |
| AdamsonGI | Full | +0.4459 ± 0.0062 | +0.1887 ± 0.0196 | +0.0444 ± 0.0304 |
| AdamsonGI | Scratch | +0.4428 ± 0.0072 | +0.1939 ± 0.0345 | +0.0614 ± 0.0340 |
| Corn | zero_shot_raw | +0.0387 ± 0.0089 | +0.0441 ± 0.0198 | +0.0585 ± 0.0347 |
| Corn | zero_shot_ols | +0.0387 ± 0.0089 | +0.0315 ± 0.0366 | +0.0390 ± 0.0557 |
| Corn | LP_last | +0.2478 ± 0.0158 | +0.0734 ± 0.0245 | +0.0171 ± 0.0378 |
| Corn | Full | +0.4640 ± 0.0035 | +0.2137 ± 0.0132 | +0.0337 ± 0.0553 |
| Corn | Scratch | +0.4587 ± 0.0033 | +0.2272 ± 0.0089 | +0.0413 ± 0.0210 |
| Gilbert | zero_shot_raw | +0.0363 ± 0.0226 | +0.1021 ± 0.0194 | +0.1700 ± 0.0343 |
| Gilbert | zero_shot_ols | +0.0363 ± 0.0226 | +0.1021 ± 0.0194 | +0.1700 ± 0.0343 |
| Gilbert | LP_last | +0.2329 ± 0.0038 | +0.1449 ± 0.0285 | +0.1364 ± 0.0591 |
| Gilbert | Full | +0.5349 ± 0.0034 | +0.2203 ± 0.0185 | +0.1695 ± 0.0348 |
| Gilbert | Scratch | +0.5225 ± 0.0068 | +0.1877 ± 0.0548 | +0.0654 ± 0.0678 |

### siamese_sl_pd_pooled — all_unfiltered (all test rows)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.0919 ± 0.0230 | -0.0993 ± 0.0501 | -0.1168 ± 0.0955 |
| Adamson | zero_shot_ols | +0.0919 ± 0.0230 | +0.0993 ± 0.0501 | +0.1168 ± 0.0955 |
| Adamson | LP_last | +0.6906 ± 0.0113 | +0.3708 ± 0.0400 | +0.2381 ± 0.0901 |
| Adamson | Full | +0.9873 ± 0.0004 | +0.6206 ± 0.0214 | +0.1701 ± 0.1431 |
| Adamson | Scratch | +0.9865 ± 0.0002 | +0.5704 ± 0.0118 | +0.0241 ± 0.1604 |
| AdamsonGI | zero_shot_raw | +0.0522 ± 0.0058 | +0.0443 ± 0.0198 | +0.0472 ± 0.0325 |
| AdamsonGI | zero_shot_ols | +0.0522 ± 0.0058 | +0.0443 ± 0.0198 | +0.0472 ± 0.0325 |
| AdamsonGI | LP_last | +0.2137 ± 0.0088 | +0.1034 ± 0.0223 | +0.0729 ± 0.0286 |
| AdamsonGI | Full | +0.4523 ± 0.0061 | +0.1939 ± 0.0202 | +0.0472 ± 0.0325 |
| AdamsonGI | Scratch | +0.4489 ± 0.0070 | +0.1991 ± 0.0358 | +0.0621 ± 0.0361 |
| Corn | zero_shot_raw | +0.0384 ± 0.0092 | +0.0438 ± 0.0198 | +0.0588 ± 0.0346 |
| Corn | zero_shot_ols | +0.0384 ± 0.0092 | +0.0314 ± 0.0363 | +0.0392 ± 0.0559 |
| Corn | LP_last | +0.2490 ± 0.0158 | +0.0753 ± 0.0247 | +0.0183 ± 0.0385 |
| Corn | Full | +0.4649 ± 0.0037 | +0.2157 ± 0.0133 | +0.0337 ± 0.0556 |
| Corn | Scratch | +0.4596 ± 0.0032 | +0.2291 ± 0.0086 | +0.0403 ± 0.0215 |
| Gilbert | zero_shot_raw | +0.0947 ± 0.0118 | +0.1001 ± 0.0059 | +0.1139 ± 0.0063 |
| Gilbert | zero_shot_ols | +0.0947 ± 0.0118 | +0.1001 ± 0.0059 | +0.1139 ± 0.0063 |
| Gilbert | LP_last | +0.2549 ± 0.0137 | +0.1564 ± 0.0289 | +0.1276 ± 0.0140 |
| Gilbert | Full | +0.5292 ± 0.0024 | +0.2856 ± 0.0095 | +0.1281 ± 0.0238 |
| Gilbert | Scratch | +0.5154 ± 0.0074 | +0.2330 ± 0.0549 | +0.0275 ± 0.0491 |

### siamese_sl — matched (test, not in SynLethDB (SL or screened non-SL), both genes in genes.txt)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.1931 ± 0.0341 | -0.1935 ± 0.0638 | -0.1968 ± 0.1054 |
| Adamson | zero_shot_ols | +0.1931 ± 0.0341 | +0.1935 ± 0.0638 | +0.1968 ± 0.1054 |
| Adamson | LP_last | +0.6950 ± 0.0377 | +0.3135 ± 0.0331 | +0.1845 ± 0.1075 |
| Adamson | Full | +0.9867 ± 0.0006 | +0.5757 ± 0.0316 | +0.2336 ± 0.1093 |
| Adamson | Scratch | +0.9861 ± 0.0004 | +0.5307 ± 0.0319 | +0.1218 ± 0.1038 |
| AdamsonGI | zero_shot_raw | +0.0526 ± 0.0140 | +0.0431 ± 0.0259 | +0.0341 ± 0.0357 |
| AdamsonGI | zero_shot_ols | +0.0526 ± 0.0140 | +0.0431 ± 0.0259 | +0.0341 ± 0.0357 |
| AdamsonGI | LP_last | +0.2163 ± 0.0210 | +0.0847 ± 0.0320 | +0.0356 ± 0.0517 |
| AdamsonGI | Full | +0.4534 ± 0.0023 | +0.2176 ± 0.0188 | +0.0375 ± 0.0411 |
| AdamsonGI | Scratch | +0.4499 ± 0.0054 | +0.2036 ± 0.0301 | +0.0881 ± 0.0217 |
| Corn | zero_shot_raw | +0.0212 ± 0.0109 | +0.0291 ± 0.0081 | +0.0319 ± 0.0247 |
| Corn | zero_shot_ols | +0.0212 ± 0.0109 | +0.0291 ± 0.0081 | +0.0319 ± 0.0247 |
| Corn | LP_last | +0.2514 ± 0.0120 | +0.0874 ± 0.0215 | +0.0544 ± 0.0444 |
| Corn | Full | +0.4799 ± 0.0036 | +0.2027 ± 0.0105 | +0.0339 ± 0.0238 |
| Corn | Scratch | +0.4803 ± 0.0042 | +0.2346 ± 0.0178 | +0.0163 ± 0.0488 |
| Gilbert | zero_shot_raw | +0.0131 ± 0.0322 | +0.0426 ± 0.0226 | +0.0925 ± 0.0730 |
| Gilbert | zero_shot_ols | +0.0131 ± 0.0322 | +0.0426 ± 0.0226 | +0.0925 ± 0.0730 |
| Gilbert | LP_last | +0.1721 ± 0.0200 | +0.1150 ± 0.0332 | +0.0694 ± 0.0720 |
| Gilbert | Full | +0.4355 ± 0.0082 | +0.1736 ± 0.0157 | +0.0704 ± 0.0363 |
| Gilbert | Scratch | +0.4217 ± 0.0046 | +0.1967 ± 0.0353 | +0.0499 ± 0.0673 |

### siamese_sl — all_sldb_filtered (test, not in SynLethDB (SL or screened non-SL; no genes.txt restriction))

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.1838 ± 0.0377 | -0.1926 ± 0.0557 | -0.2044 ± 0.0962 |
| Adamson | zero_shot_ols | +0.1838 ± 0.0377 | +0.1926 ± 0.0557 | +0.2044 ± 0.0962 |
| Adamson | LP_last | +0.6933 ± 0.0376 | +0.3253 ± 0.0456 | +0.2000 ± 0.0869 |
| Adamson | Full | +0.9865 ± 0.0006 | +0.5955 ± 0.0223 | +0.2466 ± 0.0796 |
| Adamson | Scratch | +0.9859 ± 0.0004 | +0.5512 ± 0.0211 | +0.1020 ± 0.1325 |
| AdamsonGI | zero_shot_raw | +0.0424 ± 0.0097 | +0.0332 ± 0.0213 | +0.0280 ± 0.0365 |
| AdamsonGI | zero_shot_ols | +0.0424 ± 0.0097 | +0.0332 ± 0.0213 | +0.0280 ± 0.0365 |
| AdamsonGI | LP_last | +0.1967 ± 0.0203 | +0.0752 ± 0.0283 | +0.0318 ± 0.0430 |
| AdamsonGI | Full | +0.4470 ± 0.0029 | +0.1955 ± 0.0166 | +0.0328 ± 0.0426 |
| AdamsonGI | Scratch | +0.4449 ± 0.0059 | +0.1874 ± 0.0280 | +0.0745 ± 0.0146 |
| Corn | zero_shot_raw | +0.0242 ± 0.0151 | +0.0320 ± 0.0100 | +0.0387 ± 0.0236 |
| Corn | zero_shot_ols | +0.0242 ± 0.0151 | +0.0320 ± 0.0100 | +0.0387 ± 0.0236 |
| Corn | LP_last | +0.2373 ± 0.0119 | +0.0845 ± 0.0190 | +0.0472 ± 0.0410 |
| Corn | Full | +0.4619 ± 0.0039 | +0.1960 ± 0.0067 | +0.0393 ± 0.0231 |
| Corn | Scratch | +0.4611 ± 0.0053 | +0.2249 ± 0.0153 | +0.0219 ± 0.0260 |
| Gilbert | zero_shot_raw | +0.0421 ± 0.0345 | +0.1038 ± 0.0240 | +0.1471 ± 0.0248 |
| Gilbert | zero_shot_ols | +0.0421 ± 0.0345 | +0.1038 ± 0.0240 | +0.1471 ± 0.0248 |
| Gilbert | LP_last | +0.2409 ± 0.0151 | +0.1523 ± 0.0184 | +0.1230 ± 0.0257 |
| Gilbert | Full | +0.5323 ± 0.0055 | +0.2128 ± 0.0268 | +0.1207 ± 0.0444 |
| Gilbert | Scratch | +0.5179 ± 0.0036 | +0.2307 ± 0.0220 | +0.0689 ± 0.0877 |

### siamese_sl — all_unfiltered (all test rows)

| Screen | Method | random pairs | >=1 gene unseen | both genes unseen |
|---|---|---|---|---|
| Adamson | zero_shot_raw | -0.1905 ± 0.0392 | -0.1970 ± 0.0552 | -0.2079 ± 0.0970 |
| Adamson | zero_shot_ols | +0.1905 ± 0.0392 | +0.1970 ± 0.0552 | +0.2079 ± 0.0970 |
| Adamson | LP_last | +0.7012 ± 0.0365 | +0.3328 ± 0.0478 | +0.2061 ± 0.0861 |
| Adamson | Full | +0.9866 ± 0.0005 | +0.5996 ± 0.0215 | +0.2507 ± 0.0816 |
| Adamson | Scratch | +0.9859 ± 0.0005 | +0.5571 ± 0.0208 | +0.1084 ± 0.1316 |
| AdamsonGI | zero_shot_raw | +0.0451 ± 0.0104 | +0.0375 ± 0.0221 | +0.0348 ± 0.0380 |
| AdamsonGI | zero_shot_ols | +0.0451 ± 0.0104 | +0.0375 ± 0.0221 | +0.0348 ± 0.0380 |
| AdamsonGI | LP_last | +0.2007 ± 0.0208 | +0.0775 ± 0.0282 | +0.0376 ± 0.0439 |
| AdamsonGI | Full | +0.4532 ± 0.0027 | +0.2014 ± 0.0176 | +0.0383 ± 0.0427 |
| AdamsonGI | Scratch | +0.4511 ± 0.0058 | +0.1921 ± 0.0292 | +0.0762 ± 0.0147 |
| Corn | zero_shot_raw | +0.0251 ± 0.0153 | +0.0330 ± 0.0101 | +0.0390 ± 0.0228 |
| Corn | zero_shot_ols | +0.0251 ± 0.0153 | +0.0330 ± 0.0101 | +0.0390 ± 0.0228 |
| Corn | LP_last | +0.2383 ± 0.0114 | +0.0870 ± 0.0186 | +0.0483 ± 0.0400 |
| Corn | Full | +0.4627 ± 0.0038 | +0.1979 ± 0.0067 | +0.0388 ± 0.0229 |
| Corn | Scratch | +0.4621 ± 0.0053 | +0.2267 ± 0.0157 | +0.0213 ± 0.0254 |
| Gilbert | zero_shot_raw | +0.1086 ± 0.0057 | +0.1118 ± 0.0098 | +0.1269 ± 0.0278 |
| Gilbert | zero_shot_ols | +0.1086 ± 0.0057 | +0.1118 ± 0.0098 | +0.1269 ± 0.0278 |
| Gilbert | LP_last | +0.2576 ± 0.0232 | +0.1710 ± 0.0312 | +0.1345 ± 0.0285 |
| Gilbert | Full | +0.5278 ± 0.0029 | +0.2899 ± 0.0104 | +0.1778 ± 0.0414 |
| Gilbert | Scratch | +0.5058 ± 0.0053 | +0.2915 ± 0.0049 | +0.0288 ± 0.0389 |

## Per-fold zero-shot values (matched rows)

| Model | Screen | Split | n | fold0 | fold1 | fold2 | fold3 | fold4 |
|---|---|---|---|---|---|---|---|---|
| siamese_sl_pd_pooled | Adamson | cv1 | 14,727 | -0.1509 | -0.0780 | -0.0839 | -0.0766 | -0.1099 |
| siamese_sl_pd_pooled | Adamson | cv2 | 25,782 | -0.1572 | -0.0865 | -0.0868 | -0.0285 | -0.1639 |
| siamese_sl_pd_pooled | Adamson | cv3 | 2,709 | -0.1784 | -0.1259 | -0.1298 | +0.0426 | -0.2387 |
| siamese_sl_pd_pooled | AdamsonGI | cv1 | 14,586 | +0.0690 | +0.0742 | +0.0582 | +0.0604 | +0.0762 |
| siamese_sl_pd_pooled | AdamsonGI | cv2 | 27,404 | +0.0672 | +0.0608 | +0.0209 | +0.0897 | +0.0621 |
| siamese_sl_pd_pooled | AdamsonGI | cv3 | 3,181 | +0.0421 | +0.0489 | +0.0029 | +0.1073 | +0.0505 |
| siamese_sl_pd_pooled | Corn | cv1 | 16,614 | +0.0344 | +0.0502 | +0.0231 | +0.0464 | +0.0592 |
| siamese_sl_pd_pooled | Corn | cv2 | 29,200 | +0.0132 | +0.0677 | +0.0213 | +0.0657 | +0.0408 |
| siamese_sl_pd_pooled | Corn | cv3 | 3,117 | -0.0101 | +0.1007 | +0.0133 | +0.0884 | +0.0606 |
| siamese_sl_pd_pooled | Gilbert | cv1 | 1,383 | +0.0323 | +0.0016 | -0.0182 | +0.0234 | +0.0419 |
| siamese_sl_pd_pooled | Gilbert | cv2 | 2,590 | +0.0812 | +0.0427 | +0.0003 | +0.0233 | +0.0784 |
| siamese_sl_pd_pooled | Gilbert | cv3 | 299 | +0.1262 | +0.0413 | -0.0365 | +0.0792 | +0.1659 |
| siamese_sl | Adamson | cv1 | 14,727 | -0.1947 | -0.1958 | -0.2070 | -0.1319 | -0.2362 |
| siamese_sl | Adamson | cv2 | 25,782 | -0.2841 | -0.1609 | -0.1886 | -0.0974 | -0.2363 |
| siamese_sl | Adamson | cv3 | 2,709 | -0.3823 | -0.1344 | -0.1731 | -0.0704 | -0.2239 |
| siamese_sl | AdamsonGI | cv1 | 14,586 | +0.0374 | +0.0686 | +0.0460 | +0.0703 | +0.0409 |
| siamese_sl | AdamsonGI | cv2 | 27,404 | -0.0002 | +0.0562 | +0.0334 | +0.0775 | +0.0485 |
| siamese_sl | AdamsonGI | cv3 | 3,181 | -0.0218 | +0.0335 | +0.0244 | +0.0890 | +0.0455 |
| siamese_sl | Corn | cv1 | 16,614 | +0.0118 | +0.0311 | +0.0360 | +0.0075 | +0.0199 |
| siamese_sl | Corn | cv2 | 29,200 | +0.0196 | +0.0411 | +0.0344 | +0.0211 | +0.0292 |
| siamese_sl | Corn | cv3 | 3,117 | +0.0526 | +0.0621 | +0.0169 | -0.0064 | +0.0342 |
| siamese_sl | Gilbert | cv1 | 1,383 | +0.0094 | +0.0074 | +0.0608 | -0.0387 | +0.0265 |
| siamese_sl | Gilbert | cv2 | 2,590 | +0.0592 | +0.0542 | +0.0679 | +0.0110 | +0.0204 |
| siamese_sl | Gilbert | cv3 | 299 | +0.1783 | +0.1408 | -0.0371 | +0.0976 | +0.0829 |
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
- **SLeuth variants**: siamese_sl_pd_pooled = siamese_sl/results/pca_joint/var_pd_pooled__best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3; siamese_sl = siamese_sl/results/pca_joint/best_cat_geneformer_scgpt_go_go2vec_kg_complex_ppi_svd_ppi_raw_mashup_esm1b_prot_t5_text_embed_cv3
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
- **PCA arm of the SLeuth combos**: joint (PCA on the concatenation of all modalities (variance 0.8))

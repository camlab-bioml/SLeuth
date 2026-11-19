## Table A (main, scope all_pairs): SynLethDB label vs screen phenotype, split by the SL pair's cell line

AUROC = P(target of an SL pair > target of a screened non-SL pair) + ties/2 (target = -phenotype, higher = more SL-like); r_rb = 2 AUROC - 1; d = sqrt(2) Phi^-1(AUROC) (binormal); 95% percentile CI, pair bootstrap B = 1000, seed 42. Negatives = every screened non-SL pair of the screen (the published negatives).

| Screen | SL subset | n SL / n non-SL | AUROC [95% CI] | r_rb | d | Spearman [95% CI] | p vs chance (Holm F5) |
|---|---|---|---|---|---|---|---|
| Adamson (Gamma, growth) | all SL (published) | 612 / 4,182 | 0.418 [0.393, 0.445] | -0.163 | -0.29 | -0.094 [-0.125, -0.063] | 0.028 |
| Adamson (Gamma, growth) | own-line SL | 258 / 4,182 | 0.648 [0.614, 0.679] | +0.296 | +0.54 | +0.120 [+0.092, +0.146] | 0.028 |
| Adamson (Gamma, growth) | other-line SL | 191 / 4,182 | 0.280 [0.242, 0.320] | -0.440 | -0.82 | -0.156 [-0.182, -0.125] | 0.028 |
| Adamson (Gamma, growth) | no-line SL | 163 / 4,182 | 0.217 [0.180, 0.255] | -0.567 | -1.11 | -0.187 [-0.214, -0.158] | 0.028 |
| AdamsonGI (Gamma.GI) | all SL (published) | 612 / 4,093 | 0.556 [0.531, 0.582] | +0.112 | +0.20 | +0.065 [+0.036, +0.095] | 0.028 |
| AdamsonGI (Gamma.GI) | own-line SL | 258 / 4,093 | 0.635 [0.596, 0.675] | +0.269 | +0.49 | +0.110 [+0.079, +0.145] | 0.028 |
| AdamsonGI (Gamma.GI) | other-line SL | 191 / 4,093 | 0.527 [0.490, 0.565] | +0.054 | +0.10 | +0.019 [-0.007, +0.046] | 0.316 |
| AdamsonGI (Gamma.GI) | no-line SL | 163 / 4,093 | 0.465 [0.423, 0.507] | -0.070 | -0.12 | -0.023 [-0.051, +0.004] | 0.312 |
| Corn (sens.score, RPE-1) | all SL (published) | 240 / 1,011 | 0.544 [0.505, 0.584] | +0.088 | +0.16 | +0.062 [+0.007, +0.119] | 0.104 |
| Corn (sens.score, RPE-1) | own-line SL | 0 / 1,011 | n/a | n/a | n/a | n/a | n/a |
| Corn (sens.score, RPE-1) | other-line SL | 172 / 1,011 | 0.531 [0.485, 0.580] | +0.062 | +0.11 | +0.040 [-0.019, +0.103] | 0.316 |
| Corn (sens.score, RPE-1) | no-line SL | 68 / 1,011 | 0.576 [0.512, 0.644] | +0.152 | +0.27 | +0.067 [+0.010, +0.124] | 0.060 |
| Gilbert (K562_avg) | all SL (published) | 6,021 / 81,092 | 0.742 [0.735, 0.750] | +0.485 | +0.92 | +0.213 [+0.207, +0.220] | 0.028 |
| Gilbert (K562_avg) | own-line SL | 4,076 / 81,092 | 0.849 [0.842, 0.856] | +0.698 | +1.46 | +0.258 [+0.252, +0.265] | 0.028 |
| Gilbert (K562_avg) | other-line SL | 1,944 / 81,092 | 0.519 [0.506, 0.532] | +0.038 | +0.07 | +0.010 [+0.003, +0.017] | 0.028 |
| Gilbert (K562_avg) | no-line SL | 1 / 81,092 | n/a | n/a | n/a | n/a | n/a |

## Table B (scope all_pairs): own-line minus other-line SL (paired bootstrap of the difference)

| Screen | Delta AUROC [95% CI] | p (two-sided) | p Holm (F1) | Delta Spearman [95% CI] | p | p Holm (F2) | direct AUROC own vs other [95% CI] | p |
|---|---|---|---|---|---|---|---|---|
| Adamson (Gamma, growth) | +0.368 [+0.316, +0.414] | <= 0.002 | 0.006 | +0.276 [+0.236, +0.312] | <= 0.002 | 0.006 | 0.835 [0.793, 0.872] | <= 0.002 |
| AdamsonGI (Gamma.GI) | +0.107 [+0.056, +0.159] | <= 0.002 | 0.006 | +0.091 [+0.052, +0.133] | <= 0.002 | 0.006 | 0.626 [0.576, 0.678] | <= 0.002 |
| Corn (sens.score, RPE-1) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| Gilbert (K562_avg) | +0.330 [+0.315, +0.345] | <= 0.002 | 0.006 | +0.248 [+0.239, +0.259] | <= 0.002 | 0.006 | 0.837 [0.827, 0.847] | <= 0.002 |

## Table C (scope all_pairs, secondary): breadth, SL in >= 2 cell lines vs exactly 1

| Screen | multi-line n | AUROC [95% CI] | single-line n | AUROC [95% CI] | Delta AUROC [95% CI] | p | p Holm (F3) | Delta Spearman [95% CI] | p Holm (F4) |
|---|---|---|---|---|---|---|---|---|---|
| Adamson (Gamma, growth) | 33 | 0.391 [0.307, 0.480] | 416 | 0.499 [0.470, 0.531] | -0.108 [-0.197, -0.018] | 0.026 | 0.078 | -0.033 [-0.075, +0.008] | 0.236 |
| AdamsonGI (Gamma.GI) | 33 | 0.586 [0.479, 0.689] | 416 | 0.589 [0.560, 0.619] | -0.003 [-0.108, +0.100] | 0.944 | 1.000 | -0.063 [-0.105, -0.019] | 0.008 |
| Corn (sens.score, RPE-1) | 17 | 0.565 [0.413, 0.703] | 155 | 0.527 [0.480, 0.577] | +0.038 [-0.120, +0.183] | 0.592 | 1.000 | -0.004 [-0.088, +0.081] | 0.968 |
| Gilbert (K562_avg) | 589 | 0.709 [0.686, 0.734] | 5431 | 0.746 [0.738, 0.755] | -0.037 [-0.063, -0.011] | 0.006 | 0.024 | -0.146 [-0.156, -0.135] | 0.008 |

## Table D (scope all_pairs, sensitivity / diagnostics)

| Screen | subset | n SL / n non-SL | AUROC [95% CI] | Spearman [95% CI] |
|---|---|---|---|---|
| Adamson | not own-line SL (other + no-line) | 354 / 4,182 | 0.251 [0.223, 0.280] | -0.231 [-0.258, -0.203] |
| Adamson | other-line SL, not non-SL in own line | 131 / 4,182 | 0.201 [0.160, 0.244] | -0.178 [-0.204, -0.149] |
| Adamson | other-line SL, non-SL in own line | 60 / 4,182 | 0.452 [0.390, 0.519] | -0.019 [-0.045, +0.008] |
| Adamson | own-line label (SL vs non-SL in own line) | 258 / 3,780 | 0.617 [0.581, 0.650] | +0.099 [+0.067, +0.127] |
| Adamson | SL from PMID 30033366 (Horlbeck 2018) | 53 / 4,182 | 0.673 [0.611, 0.729] | +0.067 [+0.042, +0.088] |
| Adamson | SL from PMID 31300006 (GEMINI) | 267 / 4,182 | 0.591 [0.558, 0.625] | +0.075 [+0.046, +0.103] |
| Adamson | SL from other sources | 292 / 4,182 | 0.214 [0.187, 0.243] | -0.245 [-0.271, -0.217] |
| AdamsonGI | not own-line SL (other + no-line) | 354 / 4,093 | 0.499 [0.471, 0.526] | -0.001 [-0.028, +0.024] |
| AdamsonGI | other-line SL, not non-SL in own line | 131 / 4,093 | 0.554 [0.512, 0.597] | +0.033 [+0.007, +0.058] |
| AdamsonGI | other-line SL, non-SL in own line | 60 / 4,093 | 0.468 [0.399, 0.536] | -0.013 [-0.042, +0.015] |
| AdamsonGI | own-line label (SL vs non-SL in own line) | 258 / 3,691 | 0.637 [0.599, 0.677] | +0.117 [+0.085, +0.154] |
| AdamsonGI | SL from PMID 30033366 (Horlbeck 2018) | 53 / 4,093 | 0.660 [0.580, 0.737] | +0.062 [+0.029, +0.094] |
| AdamsonGI | SL from PMID 31300006 (GEMINI) | 267 / 4,093 | 0.584 [0.547, 0.622] | +0.070 [+0.038, +0.103] |
| AdamsonGI | SL from other sources | 292 / 4,093 | 0.511 [0.481, 0.541] | +0.010 [-0.017, +0.036] |
| Corn | not own-line SL (other + no-line) | 240 / 1,011 | 0.544 [0.505, 0.584] | +0.062 [+0.007, +0.119] |
| Corn | other-line SL, not non-SL in own line | 169 / 1,011 | 0.535 [0.488, 0.585] | +0.044 [-0.015, +0.107] |
| Corn | other-line SL, non-SL in own line | 3 / 1,011 | n/a | n/a |
| Corn | own-line label (SL vs non-SL in own line) | 0 / 66 | n/a | n/a |
| Corn | SL from PMID 30033366 (Horlbeck 2018) | 19 / 1,011 | 0.601 [0.457, 0.734] | +0.050 [-0.020, +0.113] |
| Corn | SL from PMID 31300006 (GEMINI) | 115 / 1,011 | 0.531 [0.472, 0.594] | +0.034 [-0.031, +0.100] |
| Corn | SL from other sources | 106 / 1,011 | 0.548 [0.495, 0.600] | +0.051 [-0.005, +0.108] |
| Gilbert | not own-line SL (other + no-line) | 1,945 / 81,092 | 0.519 [0.506, 0.532] | +0.010 [+0.003, +0.017] |
| Gilbert | other-line SL, not non-SL in own line | 99 / 81,092 | 0.499 [0.443, 0.554] | -0.000 [-0.007, +0.006] |
| Gilbert | other-line SL, non-SL in own line | 1,845 / 81,092 | 0.520 [0.506, 0.534] | +0.010 [+0.003, +0.017] |
| Gilbert | own-line label (SL vs non-SL in own line) | 4,076 / 82,896 | 0.849 [0.842, 0.856] | +0.255 [+0.249, +0.262] |
| Gilbert | SL from PMID 30033366 (Horlbeck 2018) | 923 / 81,092 | 0.906 [0.891, 0.919] | +0.148 [+0.141, +0.155] |
| Gilbert | SL from PMID 31300006 (GEMINI) | 4,957 / 81,092 | 0.707 [0.699, 0.715] | +0.167 [+0.160, +0.174] |
| Gilbert | SL from other sources | 141 / 81,092 | 0.928 [0.897, 0.958] | +0.062 [+0.055, +0.068] |

| Screen | sensitivity contrast | Delta AUROC [95% CI] | p | p Holm |
|---|---|---|---|---|
| Adamson | own_line - not_own_line | +0.397 [+0.357, +0.438] | <= 0.002 | 0.006 |
| Adamson | own_line - other_line_uncontradicted | +0.447 [+0.393, +0.496] | <= 0.002 | 0.006 |
| Adamson | ownline_label - all | +0.198 [+0.169, +0.228] | <= 0.002 | 0.006 |
| Adamson | src_horlbeck2018 - src_gemini2019 | +0.082 [+0.016, +0.146] | 0.026 | - |
| AdamsonGI | own_line - not_own_line | +0.136 [+0.089, +0.185] | <= 0.002 | 0.006 |
| AdamsonGI | own_line - other_line_uncontradicted | +0.080 [+0.024, +0.139] | 0.008 | 0.008 |
| AdamsonGI | ownline_label - all | +0.081 [+0.055, +0.109] | <= 0.002 | 0.006 |
| AdamsonGI | src_horlbeck2018 - src_gemini2019 | +0.075 [-0.013, +0.158] | 0.090 | - |
| Corn | own_line - not_own_line | n/a | n/a | n/a |
| Corn | own_line - other_line_uncontradicted | n/a | n/a | n/a |
| Corn | ownline_label - all | n/a | n/a | n/a |
| Corn | src_horlbeck2018 - src_gemini2019 | +0.071 [-0.081, +0.210] | 0.362 | - |
| Gilbert | own_line - not_own_line | +0.330 [+0.315, +0.345] | <= 0.002 | 0.006 |
| Gilbert | own_line - other_line_uncontradicted | +0.350 [+0.295, +0.406] | <= 0.002 | 0.006 |
| Gilbert | ownline_label - all | +0.106 [+0.100, +0.112] | <= 0.002 | 0.006 |
| Gilbert | src_horlbeck2018 - src_gemini2019 | +0.199 [+0.181, +0.216] | <= 0.002 | - |

## Table E (per screen split, scope test_genes_txt): AUROC [95% CI] by subset

| Screen | split | all | own-line | other-line | no-line | Delta own-other [95% CI] (p) |
|---|---|---|---|---|---|---|
| Adamson | cv1 | 0.428 [0.372, 0.488] (n=111) | 0.683 [0.618, 0.741] (n=43) | 0.323 [0.234, 0.419] (n=28) | 0.228 [0.158, 0.309] (n=40) | +0.360 [+0.248, +0.465] (<= 0.002) |
| Adamson | cv2 | 0.399 [0.355, 0.447] (n=194) | 0.618 [0.562, 0.674] (n=99) | 0.203 [0.144, 0.263] (n=60) | 0.118 [0.062, 0.186] (n=35) | +0.416 [+0.338, +0.492] (<= 0.002) |
| Adamson | cv3 | 0.436 [0.252, 0.642] (n=14) | n/a (n=9) | n/a (n=4) | n/a (n=1) | n/a (n/a) |
| AdamsonGI | cv1 | 0.535 [0.479, 0.590] (n=119) | 0.625 [0.532, 0.713] (n=52) | 0.498 [0.412, 0.579] (n=38) | 0.421 [0.336, 0.501] (n=29) | +0.126 [+0.005, +0.248] (0.038) |
| AdamsonGI | cv2 | 0.535 [0.500, 0.568] (n=335) | 0.665 [0.604, 0.723] (n=107) | 0.508 [0.460, 0.554] (n=119) | 0.437 [0.386, 0.485] (n=109) | +0.157 [+0.081, +0.234] (<= 0.002) |
| AdamsonGI | cv3 | 0.532 [0.429, 0.621] (n=52) | 0.685 [0.556, 0.815] (n=15) | 0.497 [0.371, 0.624] (n=18) | 0.444 [0.319, 0.561] (n=19) | +0.188 [+0.031, +0.344] (0.024) |
| Corn | cv1 | 0.552 [0.466, 0.642] (n=50) | n/a (n=0) | 0.512 [0.421, 0.620] (n=38) | 0.680 [0.549, 0.801] (n=12) | n/a (n/a) |
| Corn | cv2 | 0.497 [0.434, 0.562] (n=96) | n/a (n=0) | 0.476 [0.390, 0.561] (n=57) | 0.529 [0.435, 0.613] (n=39) | n/a (n/a) |
| Corn | cv3 | n/a (n=9) | n/a (n=0) | n/a (n=4) | n/a (n=5) | n/a (n/a) |
| Gilbert | cv1 | 0.734 [0.718, 0.749] (n=1211) | 0.841 [0.824, 0.857] (n=794) | 0.531 [0.504, 0.559] (n=417) | n/a (n=0) | +0.309 [+0.276, +0.343] (<= 0.002) |
| Gilbert | cv2 | 0.713 [0.700, 0.725] (n=2368) | 0.821 [0.808, 0.833] (n=1560) | 0.505 [0.485, 0.523] (n=808) | n/a (n=0) | +0.316 [+0.292, +0.339] (<= 0.002) |
| Gilbert | cv3 | 0.677 [0.637, 0.715] (n=276) | 0.782 [0.740, 0.820] (n=174) | 0.497 [0.444, 0.559] (n=102) | n/a (n=0) | +0.284 [+0.214, +0.353] (<= 0.002) |

## Table F (scope all_pairs): same-map cell-line contrast, K562 vs Jurkat-only SL, both from PMID 30033366 / 31300006 (same CRISPRi source)

| Screen | K562 SL n | AUROC (pair CI) | Jurkat-only SL n | AUROC (pair CI) | Delta (pair CI), p, p Holm | Delta (gene CI), p, p Holm |
|---|---|---|---|---|---|---|
| Adamson (Gamma, growth) | 245 | 0.661 [0.627, 0.692] | 60 | 0.452 [0.390, 0.519] | +0.208 [+0.131, +0.276], <= 0.002, 0.006 | +0.208 [+0.035, +0.395], 0.024, 0.048 |
| AdamsonGI (Gamma.GI) | 245 | 0.632 [0.593, 0.673] | 60 | 0.468 [0.399, 0.536] | +0.164 [+0.092, +0.244], <= 0.002, 0.006 | +0.164 [-0.013, +0.289], 0.062, 0.062 |
| Corn (sens.score, RPE-1) | 0 | n/a | 55 | 0.448 [0.378, 0.522] | n/a | n/a |
| Gilbert (K562_avg) | 3966 | 0.845 [0.838, 0.853] | 1914 | 0.516 [0.502, 0.528] | +0.330 [+0.315, +0.345], <= 0.002, 0.006 | +0.330 [+0.280, +0.376], <= 0.002, 0.006 |

## Table G (scope all_pairs): pair bootstrap (stated rule) vs gene-level (vertex) bootstrap

| Screen | subset | AUROC | pair-bootstrap 95% CI | gene-bootstrap 95% CI | gene-bootstrap p vs 0.5 |
|---|---|---|---|---|---|
| Adamson | all SL (published) | 0.418 | [0.393, 0.445] | [0.311, 0.552] | 0.290 |
| Adamson | own-line SL | 0.648 | [0.614, 0.679] | [0.569, 0.725] | 0.002 |
| Adamson | other-line SL | 0.280 | [0.242, 0.320] | [0.175, 0.413] | <= 0.002 |
| Adamson | no-line SL | 0.217 | [0.180, 0.255] | [0.128, 0.407] | 0.004 |
| AdamsonGI | all SL (published) | 0.556 | [0.531, 0.582] | [0.487, 0.629] | 0.094 |
| AdamsonGI | own-line SL | 0.635 | [0.596, 0.675] | [0.528, 0.720] | 0.016 |
| AdamsonGI | other-line SL | 0.527 | [0.490, 0.565] | [0.455, 0.606] | 0.456 |
| AdamsonGI | no-line SL | 0.465 | [0.423, 0.507] | [0.390, 0.605] | 0.652 |
| Corn | all SL (published) | 0.544 | [0.505, 0.584] | [0.465, 0.621] | 0.274 |
| Corn | own-line SL | n/a | n/a | n/a | n/a |
| Corn | other-line SL | 0.531 | [0.485, 0.580] | [0.444, 0.627] | 0.508 |
| Corn | no-line SL | 0.576 | [0.512, 0.644] | [0.416, 0.704] | 0.348 |
| Gilbert | all SL (published) | 0.742 | [0.735, 0.750] | [0.695, 0.796] | <= 0.002 |
| Gilbert | own-line SL | 0.849 | [0.842, 0.856] | [0.806, 0.888] | <= 0.002 |
| Gilbert | other-line SL | 0.519 | [0.506, 0.532] | [0.481, 0.563] | 0.330 |
| Gilbert | no-line SL | n/a | n/a | n/a | n/a |

| Screen | contrast | Delta AUROC | pair CI, p (Holm) | gene CI, p (Holm) |
|---|---|---|---|---|
| Adamson | own - other | +0.368 | [+0.316, +0.414], <= 0.002 (0.006) | [+0.219, +0.492], <= 0.002 (0.006) |
| AdamsonGI | own - other | +0.107 | [+0.056, +0.159], <= 0.002 (0.006) | [-0.024, +0.210], 0.112 (0.112) |
| Corn | own - other | n/a | n/a | n/a |
| Gilbert | own - other | +0.330 | [+0.315, +0.345], <= 0.002 (0.006) | [+0.280, +0.376], <= 0.002 (0.006) |

## Table H (scope all_pairs): raw phenotype (outcome_raw; SL-like = negative) by subset, and share of pairs with both genes in the Gilbert (Horlbeck 2018) screen

| Screen | subset | n | median outcome_raw [IQR] | both genes in Gilbert screen |
|---|---|---|---|---|
| Adamson | own_line | 258 | -0.253 [-0.289, -0.216] | 0.93 |
| Adamson | other_line | 191 | -0.162 [-0.210, -0.100] | 0.31 |
| Adamson | unknown_line | 163 | -0.119 [-0.186, -0.065] | 0.00 |
| Adamson | all | 612 | -0.197 [-0.255, -0.121] | 0.49 |
| Adamson | k562_same_map | 245 | -0.255 [-0.290, -0.221] | 0.95 |
| Adamson | jurkat_only_same_map | 60 | -0.205 [-0.236, -0.175] | 1.00 |
| Adamson | other_line_not_same_map | 116 | -0.121 [-0.180, -0.040] | 0.00 |
| Adamson | nonSL (all negatives) | 4,182 | -0.216 [-0.260, -0.175] | 0.89 |
| Adamson | nonSL screened in own line | 3,720 | -0.223 [-0.266, -0.187] | 1.00 |
| AdamsonGI | own_line | 258 | -0.444 [-1.207, +0.382] | 0.93 |
| AdamsonGI | other_line | 191 | -0.004 [-0.374, +0.459] | 0.31 |
| AdamsonGI | unknown_line | 163 | +0.220 [-0.278, +0.662] | 0.00 |
| AdamsonGI | all | 612 | -0.055 [-0.681, +0.529] | 0.49 |
| AdamsonGI | k562_same_map | 245 | -0.466 [-1.208, +0.448] | 0.95 |
| AdamsonGI | jurkat_only_same_map | 60 | +0.204 [-0.304, +0.719] | 1.00 |
| AdamsonGI | other_line_not_same_map | 116 | -0.085 [-0.398, +0.315] | 0.00 |
| AdamsonGI | nonSL (all negatives) | 4,093 | +0.108 [-0.505, +0.715] | 0.89 |
| AdamsonGI | nonSL screened in own line | 3,631 | +0.126 [-0.510, +0.756] | 1.00 |
| Corn | other_line | 172 | -0.196 [-0.708, +0.000] | 0.69 |
| Corn | unknown_line | 68 | -0.268 [-0.613, +0.000] | 0.00 |
| Corn | all | 240 | -0.230 [-0.673, +0.000] | 0.49 |
| Corn | jurkat_only_same_map | 55 | +0.000 [-0.492, +0.000] | 0.98 |
| Corn | other_line_not_same_map | 38 | -0.250 [-0.406, +0.000] | 0.21 |
| Corn | nonSL (all negatives) | 1,011 | -0.083 [-0.605, +0.000] | 0.81 |
| Corn | nonSL screened in own line | 57 | +0.000 [-0.193, +0.000] | 0.00 |
| Gilbert | own_line | 4,076 | -1.914 [-3.274, -0.695] | 1.00 |
| Gilbert | other_line | 1,944 | +0.051 [-0.615, +0.746] | 1.00 |
| Gilbert | unknown_line | 1 | -0.495 [-0.495, -0.495] | 1.00 |
| Gilbert | all | 6,021 | -1.084 [-2.772, +0.013] | 1.00 |
| Gilbert | k562_same_map | 3,966 | -1.864 [-3.229, -0.663] | 1.00 |
| Gilbert | jurkat_only_same_map | 1,914 | +0.065 [-0.582, +0.752] | 1.00 |
| Gilbert | other_line_not_same_map | 30 | -1.181 [-1.964, -0.113] | 1.00 |
| Gilbert | nonSL (all negatives) | 81,092 | +0.137 [-0.549, +0.792] | 1.00 |
| Gilbert | nonSL screened in own line | 81,050 | +0.137 [-0.549, +0.791] | 1.00 |

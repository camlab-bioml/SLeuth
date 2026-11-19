### T1. All-16-embedding models vs the selected combos (graded, shared evaluator)

Mean (population SD over the 5 splits) [pair-bootstrap 95% CI of the 5-split mean, B = 1000]. Source: `sl_comparison/results[/pca_joint]/paper_tables/tab_tiers.csv` (auprg/ap/auroc _mean/_sd) and `.../bootstrap/ci_tiers.csv` (fold = mean, ci_low/ci_high).

| Arm | Tier | Model | Params (fold mean) | AUPRG | AP | AUROC |
|---|---|---|---|---|---|---|
| none | I | SLeuth pooled, all 16 embeddings | 320,106 | 0.990 (0.000) [0.990, 0.990] | 0.907 (0.003) [0.904, 0.910] | 0.962 (0.001) [0.961, 0.964] |
| none | I | SLeuth pooled (selected combo) | 161,706 | 0.990 (0.001) [0.989, 0.990] | 0.905 (0.004) [0.902, 0.908] | 0.961 (0.002) [0.960, 0.963] |
| none | I | SLeuth cell-line, all 16 embeddings | 321,321 | 0.976 (0.004) [0.975, 0.977] | 0.836 (0.014) [0.832, 0.840] | 0.923 (0.004) [0.921, 0.926] |
| none | I | SLeuth cell-line (selected combo) | 162,921 | 0.975 (0.002) [0.974, 0.976] | 0.829 (0.008) [0.825, 0.833] | 0.918 (0.007) [0.916, 0.920] |
| none | II | SLeuth pooled, all 16 embeddings | 320,106 | 0.944 (0.029) [0.943, 0.946] | 0.740 (0.031) [0.736, 0.744] | 0.867 (0.032) [0.865, 0.869] |
| none | II | SLeuth pooled (selected combo) | 156,294 | 0.943 (0.029) [0.941, 0.944] | 0.728 (0.038) [0.724, 0.731] | 0.866 (0.023) [0.864, 0.868] |
| none | II | SLeuth cell-line, all 16 embeddings | 321,321 | 0.925 (0.034) [0.923, 0.927] | 0.687 (0.031) [0.683, 0.692] | 0.839 (0.020) [0.836, 0.841] |
| none | II | SLeuth cell-line (selected combo) | 157,509 | 0.900 (0.059) [0.898, 0.903] | 0.650 (0.059) [0.646, 0.655] | 0.820 (0.024) [0.817, 0.823] |
| none | III | SLeuth pooled, all 16 embeddings | 320,106 | 0.829 (0.114) [0.816, 0.841] | 0.531 (0.090) [0.516, 0.546] | 0.750 (0.045) [0.741, 0.760] |
| none | III | SLeuth pooled (selected combo) | 158,644 | 0.867 (0.058) [0.857, 0.877] | 0.552 (0.097) [0.536, 0.568] | 0.767 (0.013) [0.758, 0.776] |
| none | III | SLeuth cell-line, all 16 embeddings | 321,321 | 0.854 (0.078) [0.841, 0.864] | 0.556 (0.097) [0.541, 0.571] | 0.756 (0.044) [0.746, 0.766] |
| none | III | SLeuth cell-line (selected combo) | 159,859 | 0.824 (0.083) [0.807, 0.836] | 0.523 (0.094) [0.507, 0.538] | 0.735 (0.058) [0.724, 0.744] |
| joint | I | SLeuth pooled, all 16 embeddings | 65,056 | 0.989 (0.000) [0.989, 0.989] | 0.901 (0.003) [0.898, 0.904] | 0.959 (0.001) [0.957, 0.960] |
| joint | I | SLeuth pooled (selected combo) | 56,469 | 0.989 (0.001) [0.988, 0.989] | 0.899 (0.006) [0.896, 0.902] | 0.958 (0.002) [0.957, 0.960] |
| joint | I | SLeuth cell-line, all 16 embeddings | 66,271 | 0.972 (0.006) [0.971, 0.973] | 0.822 (0.023) [0.818, 0.826] | 0.908 (0.015) [0.906, 0.911] |
| joint | I | SLeuth cell-line (selected combo) | 57,684 | 0.974 (0.004) [0.973, 0.975] | 0.827 (0.016) [0.823, 0.832] | 0.918 (0.006) [0.916, 0.921] |
| joint | II | SLeuth pooled, all 16 embeddings | 62,911 | 0.943 (0.028) [0.941, 0.944] | 0.730 (0.023) [0.726, 0.734] | 0.863 (0.024) [0.861, 0.866] |
| joint | II | SLeuth pooled (selected combo) | 48,727 | 0.937 (0.029) [0.936, 0.939] | 0.713 (0.031) [0.709, 0.717] | 0.856 (0.021) [0.854, 0.858] |
| joint | II | SLeuth cell-line, all 16 embeddings | 64,126 | 0.893 (0.053) [0.890, 0.896] | 0.633 (0.041) [0.629, 0.638] | 0.801 (0.024) [0.798, 0.804] |
| joint | II | SLeuth cell-line (selected combo) | 49,942 | 0.882 (0.073) [0.878, 0.885] | 0.622 (0.066) [0.618, 0.627] | 0.789 (0.043) [0.786, 0.792] |
| joint | III | SLeuth pooled, all 16 embeddings | 65,194 | 0.833 (0.110) [0.820, 0.844] | 0.525 (0.129) [0.511, 0.541] | 0.740 (0.041) [0.730, 0.749] |
| joint | III | SLeuth pooled (selected combo) | 51,955 | 0.818 (0.098) [0.804, 0.829] | 0.503 (0.111) [0.488, 0.519] | 0.738 (0.035) [0.728, 0.748] |
| joint | III | SLeuth cell-line, all 16 embeddings | 66,409 | 0.723 (0.153) [0.699, 0.742] | 0.439 (0.087) [0.425, 0.454] | 0.687 (0.036) [0.677, 0.696] |
| joint | III | SLeuth cell-line (selected combo) | 53,170 | 0.677 (0.208) [0.656, 0.696] | 0.419 (0.053) [0.404, 0.435] | 0.665 (0.044) [0.655, 0.676] |

### T1b. Selected combo minus all-16 model, same variant and arm

Pair bootstrap: `ci_tiers_diffs.csv` (model = combo, peer = all-16, fold = mean): point [95% CI], two-sided p2 = min(1, 2 max(min(p, 1-p), 1/(B+1))) from p_boot_le0, B = 1000, floor 0.002; Holm over the 2 variants per (arm, metric, tier). Fold-paired: paired t over the 5 splits (graded per-fold values), exact sign-flip p (minimum 0.0625 with 5 splits), # = splits where the combo is higher.

| Arm | Variant | Tier | Metric | Combo | All-16 | Diff [boot 95% CI] | p2 boot (Holm) | fold-paired 95% CI | p_t (Holm) | p sign-flip | # combo > all-16 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | pd:pooled | I | auprg | 0.990 | 0.990 | -0.0004 [-0.0005, -0.0002] | <= 0.002 (0.004) | [-0.0006, -0.0002] | 0.009 (0.017) | 0.0625 | 0/5 |
| none | pd:pooled | I | ap | 0.905 | 0.907 | -0.0024 [-0.0032, -0.0015] | <= 0.002 (0.004) | [-0.0040, -0.0008] | 0.015 (0.029) | 0.0625 | 0/5 |
| none | pd:pooled | I | auroc | 0.961 | 0.962 | -0.0010 [-0.0015, -0.0004] | <= 0.002 (0.004) | [-0.0017, -0.0002] | 0.022 (0.045) | 0.0625 | 0/5 |
| none | pd:pooled | II | auprg | 0.943 | 0.944 | -0.0018 [-0.0028, -0.0008] | 0.002 (0.004) | [-0.0214, +0.0178] | 0.815 (0.815) | 0.6250 | 2/5 |
| none | pd:pooled | II | ap | 0.728 | 0.740 | -0.0121 [-0.0143, -0.0100] | <= 0.002 (0.004) | [-0.0529, +0.0287] | 0.456 (0.456) | 0.5000 | 2/5 |
| none | pd:pooled | II | auroc | 0.866 | 0.867 | -0.0013 [-0.0029, +0.0003] | 0.128 (0.128) | [-0.0297, +0.0271] | 0.907 (0.907) | 0.8125 | 2/5 |
| none | pd:pooled | III | auprg | 0.867 | 0.829 | +0.0382 [+0.0296, +0.0468] | <= 0.002 (0.004) | [-0.0475, +0.1238] | 0.284 (0.284) | 0.3125 | 4/5 |
| none | pd:pooled | III | ap | 0.552 | 0.531 | +0.0214 [+0.0119, +0.0310] | <= 0.002 (0.004) | [-0.0517, +0.0946] | 0.461 (0.461) | 0.3750 | 3/5 |
| none | pd:pooled | III | auroc | 0.767 | 0.750 | +0.0170 [+0.0096, +0.0245] | <= 0.002 (0.004) | [-0.0396, +0.0736] | 0.452 (0.452) | 0.5625 | 3/5 |
| none | pd:per_head | I | auprg | 0.975 | 0.976 | -0.0013 [-0.0020, -0.0007] | <= 0.002 (0.004) | [-0.0076, +0.0050] | 0.589 (0.589) | 0.6250 | 2/5 |
| none | pd:per_head | I | ap | 0.829 | 0.836 | -0.0067 [-0.0092, -0.0046] | <= 0.002 (0.004) | [-0.0326, +0.0191] | 0.509 (0.509) | 0.5000 | 2/5 |
| none | pd:per_head | I | auroc | 0.918 | 0.923 | -0.0053 [-0.0070, -0.0037] | <= 0.002 (0.004) | [-0.0189, +0.0083] | 0.339 (0.339) | 0.4375 | 1/5 |
| none | pd:per_head | II | auprg | 0.900 | 0.925 | -0.0245 [-0.0262, -0.0229] | <= 0.002 (0.004) | [-0.0670, +0.0179] | 0.184 (0.368) | 0.0625 | 0/5 |
| none | pd:per_head | II | ap | 0.650 | 0.687 | -0.0370 [-0.0394, -0.0346] | <= 0.002 (0.004) | [-0.0794, +0.0054] | 0.072 (0.145) | 0.0625 | 0/5 |
| none | pd:per_head | II | auroc | 0.820 | 0.839 | -0.0187 [-0.0206, -0.0168] | <= 0.002 (0.004) | [-0.0484, +0.0110] | 0.155 (0.310) | 0.1250 | 1/5 |
| none | pd:per_head | III | auprg | 0.824 | 0.854 | -0.0298 [-0.0387, -0.0225] | <= 0.002 (0.004) | [-0.0446, -0.0151] | 0.005 (0.010) | 0.0625 | 0/5 |
| none | pd:per_head | III | ap | 0.523 | 0.556 | -0.0331 [-0.0415, -0.0251] | <= 0.002 (0.004) | [-0.0622, -0.0040] | 0.034 (0.068) | 0.0625 | 0/5 |
| none | pd:per_head | III | auroc | 0.735 | 0.756 | -0.0214 [-0.0298, -0.0136] | <= 0.002 (0.004) | [-0.0443, +0.0014] | 0.060 (0.119) | 0.1250 | 1/5 |
| joint | pd:pooled | I | auprg | 0.989 | 0.989 | -0.0003 [-0.0005, -0.0002] | <= 0.002 (0.004) | [-0.0007, +0.0001] | 0.088 (0.175) | 0.1250 | 1/5 |
| joint | pd:pooled | I | ap | 0.899 | 0.901 | -0.0024 [-0.0033, -0.0016] | <= 0.002 (0.004) | [-0.0055, +0.0007] | 0.095 (0.191) | 0.1875 | 1/5 |
| joint | pd:pooled | I | auroc | 0.958 | 0.959 | -0.0007 [-0.0013, -0.0001] | 0.014 (0.014) | [-0.0024, +0.0010] | 0.319 (0.319) | 0.3750 | 2/5 |
| joint | pd:pooled | II | auprg | 0.937 | 0.943 | -0.0054 [-0.0065, -0.0043] | <= 0.002 (0.004) | [-0.0214, +0.0106] | 0.401 (0.802) | 0.5000 | 2/5 |
| joint | pd:pooled | II | ap | 0.713 | 0.730 | -0.0168 [-0.0192, -0.0145] | <= 0.002 (0.004) | [-0.0513, +0.0176] | 0.247 (0.493) | 0.3125 | 2/5 |
| joint | pd:pooled | II | auroc | 0.856 | 0.863 | -0.0076 [-0.0093, -0.0060] | <= 0.002 (0.004) | [-0.0242, +0.0089] | 0.269 (0.539) | 0.3125 | 2/5 |
| joint | pd:pooled | III | auprg | 0.818 | 0.833 | -0.0152 [-0.0259, -0.0046] | 0.008 (0.008) | [-0.0962, +0.0658] | 0.630 (0.886) | 0.7500 | 3/5 |
| joint | pd:pooled | III | ap | 0.503 | 0.525 | -0.0222 [-0.0331, -0.0113] | <= 0.002 (0.004) | [-0.1069, +0.0624] | 0.506 (1.000) | 0.6250 | 3/5 |
| joint | pd:pooled | III | auroc | 0.738 | 0.740 | -0.0015 [-0.0106, +0.0075] | 0.742 (0.742) | [-0.0508, +0.0477] | 0.935 (0.935) | 0.9375 | 3/5 |
| joint | pd:per_head | I | auprg | 0.974 | 0.972 | +0.0017 [+0.0010, +0.0024] | <= 0.002 (0.004) | [-0.0033, +0.0067] | 0.393 (0.393) | 0.4375 | 4/5 |
| joint | pd:per_head | I | ap | 0.827 | 0.822 | +0.0050 [+0.0026, +0.0072] | <= 0.002 (0.004) | [-0.0131, +0.0231] | 0.486 (0.486) | 0.5000 | 3/5 |
| joint | pd:per_head | I | auroc | 0.918 | 0.908 | +0.0102 [+0.0086, +0.0120] | <= 0.002 (0.004) | [-0.0062, +0.0267] | 0.159 (0.318) | 0.1875 | 4/5 |
| joint | pd:per_head | II | auprg | 0.882 | 0.893 | -0.0114 [-0.0133, -0.0094] | <= 0.002 (0.004) | [-0.0515, +0.0288] | 0.477 (0.802) | 0.6875 | 3/5 |
| joint | pd:per_head | II | ap | 0.622 | 0.633 | -0.0112 [-0.0139, -0.0084] | <= 0.002 (0.004) | [-0.0505, +0.0280] | 0.472 (0.493) | 0.5000 | 3/5 |
| joint | pd:per_head | II | auroc | 0.789 | 0.801 | -0.0123 [-0.0149, -0.0098] | <= 0.002 (0.004) | [-0.0753, +0.0507] | 0.617 (0.617) | 0.8125 | 3/5 |
| joint | pd:per_head | III | auprg | 0.677 | 0.723 | -0.0457 [-0.0626, -0.0263] | <= 0.002 (0.004) | [-0.1948, +0.1035] | 0.443 (0.886) | 0.5625 | 1/5 |
| joint | pd:per_head | III | ap | 0.419 | 0.439 | -0.0198 [-0.0300, -0.0089] | <= 0.002 (0.004) | [-0.1048, +0.0652] | 0.553 (1.000) | 0.7500 | 2/5 |
| joint | pd:per_head | III | auroc | 0.665 | 0.687 | -0.0218 [-0.0312, -0.0115] | <= 0.002 (0.004) | [-0.0640, +0.0205] | 0.226 (0.452) | 0.3125 | 2/5 |

### T1c. All-16 models minus the strongest other rows (graded, fold-paired; Holm over the 6 non-SLeuth rows per arm x variant x metric x tier)

| Arm | Variant | Tier | Metric | vs | All-16 | Other | Diff [95% CI] | p_t (Holm-6) | p sign-flip |
|---|---|---|---|---|---|---|---|---|---|
| none | pd:pooled | II | auprg | SLMGAE | 0.944 | 0.863 | +0.081 [+0.036, +0.127] | 0.008 (0.037) | 0.0625 |
| none | pd:pooled | II | auprg | SLGNN | 0.944 | 0.915 | +0.029 [+0.007, +0.051] | 0.022 (0.067) | 0.0625 |
| none | pd:pooled | II | auprg | ELISL | 0.944 | 0.869 | +0.075 [+0.037, +0.113] | 0.005 (0.032) | 0.0625 |
| none | pd:pooled | II | auprg | BilinearProbe | 0.944 | 0.903 | +0.041 [+0.006, +0.077] | 0.031 (0.067) | 0.0625 |
| none | pd:pooled | II | auroc | SLMGAE | 0.867 | 0.765 | +0.102 [+0.079, +0.125] | <0.001 (0.002) | 0.0625 |
| none | pd:pooled | II | auroc | SLGNN | 0.867 | 0.841 | +0.027 [-0.008, +0.061] | 0.101 (0.101) | 0.1250 |
| none | pd:pooled | II | auroc | ELISL | 0.867 | 0.782 | +0.085 [+0.046, +0.124] | 0.004 (0.015) | 0.0625 |
| none | pd:pooled | II | auroc | BilinearProbe | 0.867 | 0.834 | +0.033 [+0.016, +0.050] | 0.005 (0.016) | 0.0625 |
| none | pd:pooled | III | auprg | SLMGAE | 0.829 | 0.703 | +0.126 [-0.063, +0.315] | 0.137 (0.274) | 0.1250 |
| none | pd:pooled | III | auprg | SLGNN | 0.829 | 0.672 | +0.157 [+0.059, +0.255] | 0.011 (0.046) | 0.0625 |
| none | pd:pooled | III | auprg | ELISL | 0.829 | 0.756 | +0.073 [-0.088, +0.234] | 0.275 (0.275) | 0.3125 |
| none | pd:pooled | III | auprg | BilinearProbe | 0.829 | 0.799 | +0.030 [-0.005, +0.064] | 0.073 (0.220) | 0.0625 |
| none | pd:pooled | III | auroc | SLMGAE | 0.750 | 0.696 | +0.054 [-0.054, +0.162] | 0.240 (0.719) | 0.3125 |
| none | pd:pooled | III | auroc | SLGNN | 0.750 | 0.692 | +0.058 [+0.003, +0.113] | 0.042 (0.169) | 0.0625 |
| none | pd:pooled | III | auroc | ELISL | 0.750 | 0.715 | +0.035 [-0.050, +0.120] | 0.319 (0.719) | 0.3125 |
| none | pd:pooled | III | auroc | BilinearProbe | 0.750 | 0.741 | +0.009 [-0.021, +0.039] | 0.464 (0.719) | 0.4375 |
| none | pd:per_head | II | auprg | SLMGAE | 0.925 | 0.863 | +0.062 [+0.019, +0.105] | 0.016 (0.064) | 0.0625 |
| none | pd:per_head | II | auprg | SLGNN | 0.925 | 0.915 | +0.010 [-0.015, +0.035] | 0.344 (0.344) | 0.5000 |
| none | pd:per_head | II | auprg | ELISL | 0.925 | 0.869 | +0.056 [+0.033, +0.078] | 0.002 (0.014) | 0.0625 |
| none | pd:per_head | II | auprg | BilinearProbe | 0.925 | 0.903 | +0.022 [-0.008, +0.052] | 0.110 (0.220) | 0.0625 |
| none | pd:per_head | II | auroc | SLMGAE | 0.839 | 0.765 | +0.073 [+0.059, +0.087] | <0.001 (<0.001) | 0.0625 |
| none | pd:per_head | II | auroc | SLGNN | 0.839 | 0.841 | -0.002 [-0.017, +0.013] | 0.725 (1.000) | 0.7500 |
| none | pd:per_head | II | auroc | ELISL | 0.839 | 0.782 | +0.056 [+0.038, +0.075] | 0.001 (0.005) | 0.0625 |
| none | pd:per_head | II | auroc | BilinearProbe | 0.839 | 0.834 | +0.005 [-0.025, +0.035] | 0.683 (1.000) | 0.6250 |
| none | pd:per_head | III | auprg | SLMGAE | 0.854 | 0.703 | +0.151 [+0.017, +0.284] | 0.035 (0.086) | 0.0625 |
| none | pd:per_head | III | auprg | SLGNN | 0.854 | 0.672 | +0.181 [+0.083, +0.280] | 0.007 (0.028) | 0.0625 |
| none | pd:per_head | III | auprg | ELISL | 0.854 | 0.756 | +0.098 [-0.021, +0.216] | 0.084 (0.086) | 0.1250 |
| none | pd:per_head | III | auprg | BilinearProbe | 0.854 | 0.799 | +0.054 [+0.009, +0.099] | 0.029 (0.086) | 0.0625 |
| none | pd:per_head | III | auroc | SLMGAE | 0.756 | 0.696 | +0.060 [+0.005, +0.114] | 0.038 (0.114) | 0.0625 |
| none | pd:per_head | III | auroc | SLGNN | 0.756 | 0.692 | +0.064 [+0.016, +0.112] | 0.020 (0.081) | 0.1250 |
| none | pd:per_head | III | auroc | ELISL | 0.756 | 0.715 | +0.041 [-0.023, +0.104] | 0.148 (0.296) | 0.2500 |
| none | pd:per_head | III | auroc | BilinearProbe | 0.756 | 0.741 | +0.015 [-0.033, +0.062] | 0.436 (0.436) | 0.4375 |
| joint | pd:pooled | II | auprg | SLMGAE | 0.943 | 0.863 | +0.080 [+0.033, +0.126] | 0.009 (0.039) | 0.0625 |
| joint | pd:pooled | II | auprg | SLGNN | 0.943 | 0.915 | +0.027 [+0.006, +0.049] | 0.024 (0.071) | 0.0625 |
| joint | pd:pooled | II | auprg | ELISL | 0.943 | 0.869 | +0.073 [+0.036, +0.111] | 0.006 (0.033) | 0.0625 |
| joint | pd:pooled | II | auprg | BilinearProbe | 0.943 | 0.903 | +0.040 [+0.002, +0.078] | 0.042 (0.085) | 0.0625 |
| joint | pd:pooled | II | auroc | SLMGAE | 0.863 | 0.765 | +0.098 [+0.084, +0.113] | <0.001 (<0.001) | 0.0625 |
| joint | pd:pooled | II | auroc | SLGNN | 0.863 | 0.841 | +0.023 [-0.001, +0.047] | 0.059 (0.059) | 0.1250 |
| joint | pd:pooled | II | auroc | ELISL | 0.863 | 0.782 | +0.081 [+0.051, +0.112] | 0.002 (0.007) | 0.0625 |
| joint | pd:pooled | II | auroc | BilinearProbe | 0.863 | 0.834 | +0.030 [+0.010, +0.049] | 0.013 (0.026) | 0.0625 |
| joint | pd:pooled | III | auprg | SLMGAE | 0.833 | 0.703 | +0.130 [-0.089, +0.349] | 0.174 (0.523) | 0.1875 |
| joint | pd:pooled | III | auprg | SLGNN | 0.833 | 0.672 | +0.160 [-0.009, +0.330] | 0.058 (0.234) | 0.0625 |
| joint | pd:pooled | III | auprg | ELISL | 0.833 | 0.756 | +0.077 [-0.070, +0.224] | 0.221 (0.523) | 0.3125 |
| joint | pd:pooled | III | auprg | BilinearProbe | 0.833 | 0.799 | +0.034 [-0.084, +0.151] | 0.471 (0.523) | 0.5625 |
| joint | pd:pooled | III | auroc | SLMGAE | 0.740 | 0.696 | +0.043 [-0.059, +0.146] | 0.305 (0.826) | 0.3750 |
| joint | pd:pooled | III | auroc | SLGNN | 0.740 | 0.692 | +0.048 [-0.018, +0.114] | 0.113 (0.452) | 0.1875 |
| joint | pd:pooled | III | auroc | ELISL | 0.740 | 0.715 | +0.025 [-0.030, +0.079] | 0.275 (0.826) | 0.3125 |
| joint | pd:pooled | III | auroc | BilinearProbe | 0.740 | 0.741 | -0.001 [-0.071, +0.068] | 0.957 (0.957) | 0.9375 |
| joint | pd:per_head | II | auprg | SLMGAE | 0.893 | 0.863 | +0.030 [-0.005, +0.065] | 0.076 (0.382) | 0.1250 |
| joint | pd:per_head | II | auprg | SLGNN | 0.893 | 0.915 | -0.022 [-0.060, +0.016] | 0.179 (0.382) | 0.0625 |
| joint | pd:per_head | II | auprg | ELISL | 0.893 | 0.869 | +0.024 [-0.007, +0.055] | 0.101 (0.382) | 0.1875 |
| joint | pd:per_head | II | auprg | BilinearProbe | 0.893 | 0.903 | -0.010 [-0.032, +0.012] | 0.284 (0.382) | 0.2500 |
| joint | pd:per_head | II | auroc | SLMGAE | 0.801 | 0.765 | +0.036 [+0.023, +0.048] | 0.001 (0.009) | 0.0625 |
| joint | pd:per_head | II | auroc | SLGNN | 0.801 | 0.841 | -0.040 [-0.058, -0.021] | 0.004 (0.017) | 0.0625 |
| joint | pd:per_head | II | auroc | ELISL | 0.801 | 0.782 | +0.019 [-0.011, +0.048] | 0.154 (0.154) | 0.1250 |
| joint | pd:per_head | II | auroc | BilinearProbe | 0.801 | 0.834 | -0.033 [-0.055, -0.011] | 0.013 (0.040) | 0.0625 |
| joint | pd:per_head | III | auprg | SLMGAE | 0.723 | 0.703 | +0.020 [-0.191, +0.232] | 0.805 (1.000) | 0.8125 |
| joint | pd:per_head | III | auprg | SLGNN | 0.723 | 0.672 | +0.051 [-0.058, +0.159] | 0.264 (0.793) | 0.2500 |
| joint | pd:per_head | III | auprg | ELISL | 0.723 | 0.756 | -0.033 [-0.203, +0.137] | 0.618 (1.000) | 0.6875 |
| joint | pd:per_head | III | auprg | BilinearProbe | 0.723 | 0.799 | -0.076 [-0.214, +0.061] | 0.198 (0.793) | 0.3125 |
| joint | pd:per_head | III | auroc | SLMGAE | 0.687 | 0.696 | -0.009 [-0.113, +0.094] | 0.815 (1.000) | 0.8750 |
| joint | pd:per_head | III | auroc | SLGNN | 0.687 | 0.692 | -0.005 [-0.047, +0.038] | 0.770 (1.000) | 0.9375 |
| joint | pd:per_head | III | auroc | ELISL | 0.687 | 0.715 | -0.028 [-0.113, +0.057] | 0.410 (1.000) | 0.3750 |
| joint | pd:per_head | III | auroc | BilinearProbe | 0.687 | 0.741 | -0.054 [-0.119, +0.010] | 0.080 (0.319) | 0.0625 |

### T2. No PCA (arm none) minus PCA of the concatenation at 0.8 variance (arm joint), fold-paired, graded

Same modalities in both arms (tab_pca `same_modalities_as_reference` = True). Diff = none - joint [paired-t 95% CI]; Holm over the 5 SLeuth rows per (metric, tier); # = splits where no PCA is higher. Source: `s1_pca_paired.csv` from `sl_comparison/results[/pca_joint]/<row>_<cv>.json` summary.folds.

| Row | Tier | Metric | No PCA | PCA | Diff [95% CI] | p_t | Holm | p sign-flip | # |
|---|---|---|---|---|---|---|---|---|---|
| SLeuth pooled (selected combo) | I | auprg | 0.990 | 0.989 | +0.001 [+0.000, +0.002] | 0.006 | 0.025 | 0.0625 | 5/5 |
| SLeuth pooled (selected combo) | I | ap | 0.905 | 0.899 | +0.006 [+0.003, +0.009] | 0.003 | 0.014 | 0.0625 | 5/5 |
| SLeuth pooled (selected combo) | I | auroc | 0.961 | 0.958 | +0.003 [+0.001, +0.005] | 0.008 | 0.033 | 0.0625 | 5/5 |
| SLeuth pooled (selected combo) | II | auprg | 0.943 | 0.937 | +0.005 [-0.001, +0.011] | 0.080 | 0.271 | 0.1250 | 4/5 |
| SLeuth pooled (selected combo) | II | ap | 0.728 | 0.713 | +0.014 [+0.001, +0.027] | 0.038 | 0.076 | 0.0625 | 5/5 |
| SLeuth pooled (selected combo) | II | auroc | 0.866 | 0.856 | +0.010 [+0.003, +0.017] | 0.020 | 0.079 | 0.0625 | 5/5 |
| SLeuth pooled (selected combo) | III | auprg | 0.867 | 0.818 | +0.050 [-0.019, +0.119] | 0.114 | 0.380 | 0.1875 | 4/5 |
| SLeuth pooled (selected combo) | III | ap | 0.552 | 0.503 | +0.049 [-0.037, +0.135] | 0.187 | 0.375 | 0.1875 | 4/5 |
| SLeuth pooled (selected combo) | III | auroc | 0.767 | 0.738 | +0.029 [-0.006, +0.063] | 0.083 | 0.242 | 0.1250 | 4/5 |
| SLeuth cell-line (selected combo) | I | auprg | 0.975 | 0.974 | +0.001 [-0.003, +0.005] | 0.469 | 0.938 | 0.5625 | 3/5 |
| SLeuth cell-line (selected combo) | I | ap | 0.829 | 0.827 | +0.002 [-0.016, +0.020] | 0.761 | 1.000 | 0.7500 | 3/5 |
| SLeuth cell-line (selected combo) | I | auroc | 0.918 | 0.918 | -0.000 [-0.004, +0.003] | 0.722 | 0.722 | 0.6875 | 2/5 |
| SLeuth cell-line (selected combo) | II | auprg | 0.900 | 0.882 | +0.019 [-0.002, +0.040] | 0.068 | 0.271 | 0.0625 | 5/5 |
| SLeuth cell-line (selected combo) | II | ap | 0.650 | 0.622 | +0.028 [+0.018, +0.038] | 0.002 | 0.007 | 0.0625 | 5/5 |
| SLeuth cell-line (selected combo) | II | auroc | 0.820 | 0.789 | +0.031 [-0.003, +0.065] | 0.064 | 0.191 | 0.0625 | 5/5 |
| SLeuth cell-line (selected combo) | III | auprg | 0.824 | 0.677 | +0.146 [-0.040, +0.333] | 0.095 | 0.380 | 0.1250 | 4/5 |
| SLeuth cell-line (selected combo) | III | ap | 0.523 | 0.419 | +0.103 [+0.030, +0.177] | 0.017 | 0.087 | 0.0625 | 5/5 |
| SLeuth cell-line (selected combo) | III | auroc | 0.735 | 0.665 | +0.069 [-0.001, +0.140] | 0.053 | 0.242 | 0.1250 | 4/5 |
| SLeuth pooled, all 16 embeddings | I | auprg | 0.990 | 0.989 | +0.001 [+0.001, +0.001] | <0.001 | 0.002 | 0.0625 | 5/5 |
| SLeuth pooled, all 16 embeddings | I | ap | 0.907 | 0.901 | +0.006 [+0.004, +0.008] | 0.001 | 0.005 | 0.0625 | 5/5 |
| SLeuth pooled, all 16 embeddings | I | auroc | 0.962 | 0.959 | +0.004 [+0.003, +0.004] | <0.001 | <0.001 | 0.0625 | 5/5 |
| SLeuth pooled, all 16 embeddings | II | auprg | 0.944 | 0.943 | +0.002 [-0.004, +0.007] | 0.485 | 0.485 | 0.5625 | 3/5 |
| SLeuth pooled, all 16 embeddings | II | ap | 0.740 | 0.730 | +0.010 [-0.007, +0.026] | 0.193 | 0.193 | 0.2500 | 4/5 |
| SLeuth pooled, all 16 embeddings | II | auroc | 0.867 | 0.863 | +0.004 [-0.011, +0.018] | 0.535 | 0.535 | 0.5625 | 3/5 |
| SLeuth pooled, all 16 embeddings | III | auprg | 0.829 | 0.833 | -0.004 [-0.127, +0.120] | 0.941 | 0.941 | 1.0000 | 4/5 |
| SLeuth pooled, all 16 embeddings | III | ap | 0.531 | 0.525 | +0.005 [-0.083, +0.093] | 0.873 | 0.873 | 1.0000 | 4/5 |
| SLeuth pooled, all 16 embeddings | III | auroc | 0.750 | 0.740 | +0.010 [-0.074, +0.094] | 0.752 | 0.752 | 0.8750 | 4/5 |
| SLeuth cell-line, all 16 embeddings | I | auprg | 0.976 | 0.972 | +0.004 [-0.003, +0.011] | 0.183 | 0.550 | 0.1875 | 4/5 |
| SLeuth cell-line, all 16 embeddings | I | ap | 0.836 | 0.822 | +0.014 [-0.015, +0.043] | 0.257 | 0.770 | 0.3125 | 4/5 |
| SLeuth cell-line, all 16 embeddings | I | auroc | 0.923 | 0.908 | +0.015 [-0.006, +0.036] | 0.119 | 0.302 | 0.1250 | 4/5 |
| SLeuth cell-line, all 16 embeddings | II | auprg | 0.925 | 0.893 | +0.032 [+0.003, +0.061] | 0.037 | 0.186 | 0.0625 | 5/5 |
| SLeuth cell-line, all 16 embeddings | II | ap | 0.687 | 0.633 | +0.054 [+0.035, +0.073] | 0.001 | 0.007 | 0.0625 | 5/5 |
| SLeuth cell-line, all 16 embeddings | II | auroc | 0.839 | 0.801 | +0.038 [+0.015, +0.061] | 0.010 | 0.052 | 0.0625 | 5/5 |
| SLeuth cell-line, all 16 embeddings | III | auprg | 0.854 | 0.723 | +0.131 [-0.020, +0.281] | 0.074 | 0.369 | 0.0625 | 5/5 |
| SLeuth cell-line, all 16 embeddings | III | ap | 0.556 | 0.439 | +0.117 [+0.013, +0.220] | 0.035 | 0.106 | 0.0625 | 5/5 |
| SLeuth cell-line, all 16 embeddings | III | auroc | 0.756 | 0.687 | +0.069 [-0.011, +0.149] | 0.075 | 0.242 | 0.1250 | 4/5 |
| SLeuth cell-line, cell_emb zeroed | I | auprg | 0.974 | 0.974 | +0.000 [-0.004, +0.004] | 0.961 | 0.961 | 1.0000 | 3/5 |
| SLeuth cell-line, cell_emb zeroed | I | ap | 0.825 | 0.827 | -0.002 [-0.022, +0.017] | 0.767 | 1.000 | 0.8125 | 3/5 |
| SLeuth cell-line, cell_emb zeroed | I | auroc | 0.915 | 0.918 | -0.003 [-0.008, +0.001] | 0.101 | 0.302 | 0.1875 | 1/5 |
| SLeuth cell-line, cell_emb zeroed | II | auprg | 0.902 | 0.883 | +0.019 [-0.006, +0.043] | 0.101 | 0.271 | 0.0625 | 5/5 |
| SLeuth cell-line, cell_emb zeroed | II | ap | 0.651 | 0.623 | +0.027 [+0.014, +0.041] | 0.005 | 0.015 | 0.0625 | 5/5 |
| SLeuth cell-line, cell_emb zeroed | II | auroc | 0.820 | 0.793 | +0.026 [-0.004, +0.057] | 0.075 | 0.191 | 0.1250 | 4/5 |
| SLeuth cell-line, cell_emb zeroed | III | auprg | 0.841 | 0.695 | +0.146 [-0.042, +0.333] | 0.097 | 0.380 | 0.1250 | 4/5 |
| SLeuth cell-line, cell_emb zeroed | III | ap | 0.533 | 0.425 | +0.108 [+0.027, +0.189] | 0.021 | 0.087 | 0.0625 | 5/5 |
| SLeuth cell-line, cell_emb zeroed | III | auroc | 0.743 | 0.670 | +0.073 [+0.001, +0.145] | 0.048 | 0.242 | 0.1250 | 4/5 |

### T2c. Compact (paper-sized) version of T2: AUPRG only, the four multimodal models

| Model | Tier | No PCA | PCA 0.8 | Diff [95% CI] | p_t (Holm-5) | p sign-flip | splits no PCA higher |
|---|---|---|---|---|---|---|---|
| SLeuth pooled (selected combo) | I | 0.990 (0.001) | 0.989 (0.001) | +0.001 [+0.000, +0.002] | 0.006 (0.025) | 0.0625 | 5/5 |
| SLeuth pooled (selected combo) | II | 0.943 (0.029) | 0.937 (0.029) | +0.005 [-0.001, +0.011] | 0.080 (0.271) | 0.1250 | 4/5 |
| SLeuth pooled (selected combo) | III | 0.867 (0.058) | 0.818 (0.098) | +0.050 [-0.019, +0.119] | 0.114 (0.380) | 0.1875 | 4/5 |
| SLeuth cell-line (selected combo) | I | 0.975 (0.002) | 0.974 (0.004) | +0.001 [-0.003, +0.005] | 0.469 (0.938) | 0.5625 | 3/5 |
| SLeuth cell-line (selected combo) | II | 0.900 (0.059) | 0.882 (0.073) | +0.019 [-0.002, +0.040] | 0.068 (0.271) | 0.0625 | 5/5 |
| SLeuth cell-line (selected combo) | III | 0.824 (0.083) | 0.677 (0.208) | +0.146 [-0.040, +0.333] | 0.095 (0.380) | 0.1250 | 4/5 |
| SLeuth pooled, all 16 embeddings | I | 0.990 (0.000) | 0.989 (0.000) | +0.001 [+0.001, +0.001] | <0.001 (0.002) | 0.0625 | 5/5 |
| SLeuth pooled, all 16 embeddings | II | 0.944 (0.029) | 0.943 (0.028) | +0.002 [-0.004, +0.007] | 0.485 (0.485) | 0.5625 | 3/5 |
| SLeuth pooled, all 16 embeddings | III | 0.829 (0.114) | 0.833 (0.110) | -0.004 [-0.127, +0.120] | 0.941 (0.941) | 1.0000 | 4/5 |
| SLeuth cell-line, all 16 embeddings | I | 0.976 (0.004) | 0.972 (0.006) | +0.004 [-0.003, +0.011] | 0.183 (0.550) | 0.1875 | 4/5 |
| SLeuth cell-line, all 16 embeddings | II | 0.925 (0.034) | 0.893 (0.053) | +0.032 [+0.003, +0.061] | 0.037 (0.186) | 0.0625 | 5/5 |
| SLeuth cell-line, all 16 embeddings | III | 0.854 (0.078) | 0.723 (0.153) | +0.131 [-0.020, +0.281] | 0.074 (0.369) | 0.0625 | 5/5 |

Per-split Tier III AUPRG (no PCA | PCA), split 0-4:

- SLeuth pooled (selected combo): 0.9208, 0.8136, 0.7855, 0.8903, 0.9269 | 0.9101, 0.7308, 0.6725, 0.9139, 0.8605
- SLeuth cell-line (selected combo): 0.8532, 0.6967, 0.7687, 0.8615, 0.9384 | 0.8548, 0.4167, 0.4342, 0.7895, 0.8912
- SLeuth pooled, all 16 embeddings: 0.9196, 0.6743, 0.7073, 0.9187, 0.9264 | 0.9038, 0.8468, 0.6163, 0.8999, 0.8970
- SLeuth cell-line, all 16 embeddings: 0.8912, 0.7353, 0.7929, 0.8985, 0.9498 | 0.8724, 0.6551, 0.4842, 0.6984, 0.9047

### T3. Single-embedding models (self-scored AUPRG_any, mean (population SD) [rank among the 16])

Native dimension, no PCA (shared by both arms). Source: `sl_comparison/results/paper_tables/tab_solo.csv` (auprg_any_mean/_sd; identical to the runs' results.json to 1e-16), pooled = `siamese_sl/results/pcavar0.8_<emb>_<cv>`, cell-line = `siamese_sl/results/var_pd_per_head__pcavar0.8_<emb>_<cv>`. Bold-worthy: rank 1.

| Category | Embedding | Pooled I | Pooled II | Pooled III | Cell-line I | Cell-line II | Cell-line III |
|---|---|---|---|---|---|---|---|
| expression | geneformer | 0.987 (0.001) [3] | 0.821 (0.130) [15] | 0.378 (0.135) [14] | 0.968 (0.002) [4] | 0.790 (0.127) [16] | 0.482 (0.209) [15] |
| expression | scgpt | 0.987 (0.001) [5] | 0.839 (0.105) [14] | 0.302 (0.222) [15] | 0.962 (0.003) [12] | 0.812 (0.115) [14] | 0.580 (0.126) [13] |
| expression | gene2vec | 0.985 (0.001) [11] | 0.815 (0.118) [16] | 0.197 (0.162) [16] | 0.945 (0.006) [15] | 0.800 (0.108) [15] | 0.531 (0.283) [14] |
| protein_seq | esm2 | 0.986 (0.001) [9] | 0.923 (0.040) [9] | 0.847 (0.056) [5] | 0.957 (0.005) [13] | 0.885 (0.066) [7] | 0.818 (0.085) [5] |
| protein_seq | esm1b | 0.986 (0.001) [6] | 0.918 (0.045) [11] | 0.782 (0.158) [11] | 0.969 (0.003) [3] | 0.879 (0.082) [9] | 0.782 (0.092) [10] |
| protein_seq | prot_t5 | 0.986 (0.001) [8] | 0.934 (0.028) [5] | 0.852 (0.049) [4] | 0.966 (0.003) [9] | 0.908 (0.047) [2] | 0.822 (0.093) [4] |
| text | text_embed | 0.987 (0.001) [2] | 0.946 (0.021) [1] | 0.876 (0.063) [3] | 0.970 (0.002) [2] | 0.900 (0.050) [4] | 0.811 (0.089) [6] |
| text | bioconceptvec | 0.985 (0.001) [12] | 0.931 (0.031) [6] | 0.830 (0.062) [9] | 0.965 (0.002) [10] | 0.899 (0.051) [5] | 0.850 (0.055) [1] |
| ppi | node2vec_ppi | 0.985 (0.001) [10] | 0.937 (0.029) [3] | 0.881 (0.044) [2] | 0.967 (0.004) [5] | 0.919 (0.021) [1] | 0.836 (0.099) [2] |
| ppi | ppi_svd | 0.983 (0.001) [15] | 0.926 (0.037) [7] | 0.812 (0.074) [10] | 0.953 (0.008) [14] | 0.857 (0.075) [11] | 0.710 (0.148) [11] |
| ppi | ppi_raw | 0.987 (0.001) [4] | 0.922 (0.039) [10] | 0.762 (0.110) [12] | 0.967 (0.003) [7] | 0.849 (0.096) [12] | 0.695 (0.205) [12] |
| ppi | mashup | 0.986 (0.001) [7] | 0.936 (0.030) [4] | 0.838 (0.067) [7] | 0.967 (0.002) [6] | 0.882 (0.057) [8] | 0.791 (0.073) [9] |
| go | go | 0.984 (0.000) [14] | 0.924 (0.030) [8] | 0.832 (0.038) [8] | 0.966 (0.003) [8] | 0.888 (0.056) [6] | 0.822 (0.030) [3] |
| go | go2vec | 0.985 (0.001) [13] | 0.940 (0.016) [2] | 0.883 (0.022) [1] | 0.964 (0.004) [11] | 0.904 (0.042) [3] | 0.810 (0.078) [7] |
| go | onto2vec | 0.974 (0.002) [16] | 0.917 (0.032) [12] | 0.844 (0.058) [6] | 0.942 (0.004) [16] | 0.874 (0.040) [10] | 0.796 (0.093) [8] |
| kg | kg_complex | 0.988 (0.000) [1] | 0.872 (0.076) [13] | 0.432 (0.091) [13] | 0.974 (0.003) [1] | 0.847 (0.059) [13] | 0.379 (0.265) [16] |
| multimodal | selected combo, no PCA | 0.990 (0.001) | 0.943 (0.029) | 0.867 (0.058) | 0.975 (0.002) | 0.900 (0.059) | 0.824 (0.083) |
| multimodal | all 16 embeddings, no PCA | 0.990 (0.000) | 0.944 (0.029) | 0.829 (0.114) | 0.976 (0.004) | 0.925 (0.034) | 0.854 (0.078) |
| multimodal | selected combo, PCA 0.8 | 0.989 (0.001) | 0.937 (0.029) | 0.818 (0.098) | 0.974 (0.004) | 0.882 (0.073) | 0.677 (0.208) |
| multimodal | all 16 embeddings, PCA 0.8 | 0.989 (0.000) | 0.943 (0.028) | 0.833 (0.110) | 0.972 (0.006) | 0.893 (0.053) | 0.723 (0.153) |

Category means of the members (the paper's 'strongest category' sentence):

| Variant | Tier | expression | go | kg | ppi | protein_seq | text |
|---|---|---|---|---|---|---|---|
| pd:per_head | I | 0.9583 | 0.9571 | 0.9742 | 0.9635 | 0.9638 | 0.9677 |
| pd:per_head | II | 0.8005 | 0.8885 | 0.8466 | 0.8767 | 0.8907 | 0.8995 |
| pd:per_head | III | 0.5307 | 0.8094 | 0.3793 | 0.7582 | 0.8074 | 0.8305 |
| pd:pooled | I | 0.9865 | 0.9808 | 0.9877 | 0.9855 | 0.9861 | 0.9860 |
| pd:pooled | II | 0.8253 | 0.9271 | 0.8722 | 0.9301 | 0.9251 | 0.9385 |
| pd:pooled | III | 0.2922 | 0.8529 | 0.4320 | 0.8233 | 0.8268 | 0.8529 |

### T4. How many of the 16 single embeddings beat each multimodal model (fold-paired self-scored AUPRG; Holm over the 16)

| Variant | Tier | Multimodal model | Its AUPRG | # above (point) | # above, Holm p<0.05 | # below, Holm p<0.05 | best single: diff [95% CI], p_t, sign-flip |
|---|---|---|---|---|---|---|---|
| pd:pooled | I | selected combo, no PCA | 0.990 | 0 | 0 | 16 | kg_complex -0.002 [-0.0025, -0.0016], <0.001, 0.062 |
| pd:pooled | I | all-16, no PCA | 0.990 | 0 | 0 | 16 | kg_complex -0.002 [-0.0028, -0.0020], <0.001, 0.062 |
| pd:pooled | I | selected combo, PCA | 0.989 | 0 | 0 | 16 | kg_complex -0.001 [-0.0015, -0.0005], 0.005, 0.062 |
| pd:pooled | I | all-16, PCA | 0.989 | 0 | 0 | 16 | kg_complex -0.001 [-0.0016, -0.0011], <0.001, 0.062 |
| pd:pooled | II | selected combo, no PCA | 0.943 | 1 | 0 | 0 | text_embed +0.004 [-0.0217, 0.0289], 0.711, 0.938 |
| pd:pooled | II | all-16, no PCA | 0.944 | 1 | 0 | 0 | text_embed +0.002 [-0.0129, 0.0166], 0.744, 0.812 |
| pd:pooled | II | selected combo, PCA | 0.937 | 2 | 0 | 0 | text_embed +0.009 [-0.0123, 0.0301], 0.310, 0.312 |
| pd:pooled | II | all-16, PCA | 0.943 | 1 | 0 | 0 | text_embed +0.003 [-0.0072, 0.0141], 0.418, 0.500 |
| pd:pooled | III | selected combo, no PCA | 0.867 | 3 | 0 | 4 | go2vec +0.015 [-0.0582, 0.0889], 0.593, 0.625 |
| pd:pooled | III | all-16, no PCA | 0.829 | 9 | 0 | 4 | go2vec +0.054 [-0.1005, 0.2076], 0.389, 0.500 |
| pd:pooled | III | selected combo, PCA | 0.818 | 9 | 0 | 4 | go2vec +0.065 [-0.0579, 0.1883], 0.215, 0.375 |
| pd:pooled | III | all-16, PCA | 0.833 | 7 | 0 | 4 | go2vec +0.050 [-0.0813, 0.1814], 0.350, 0.438 |
| pd:per_head | I | selected combo, no PCA | 0.975 | 0 | 0 | 5 | kg_complex -0.001 [-0.0058, 0.0045], 0.740, 0.750 |
| pd:per_head | I | all-16, no PCA | 0.976 | 0 | 0 | 10 | kg_complex -0.002 [-0.0076, 0.0036], 0.381, 0.438 |
| pd:per_head | I | selected combo, PCA | 0.974 | 1 | 0 | 3 | kg_complex +0.000 [-0.0045, 0.0054], 0.801, 0.812 |
| pd:per_head | I | all-16, PCA | 0.972 | 1 | 0 | 4 | kg_complex +0.002 [-0.0023, 0.0067], 0.244, 0.312 |
| pd:per_head | II | selected combo, no PCA | 0.900 | 3 | 0 | 1 | node2vec_ppi +0.018 [-0.0420, 0.0786], 0.448, 0.625 |
| pd:per_head | II | all-16, no PCA | 0.925 | 0 | 0 | 0 | node2vec_ppi -0.006 [-0.0262, 0.0137], 0.432, 0.438 |
| pd:per_head | II | selected combo, PCA | 0.882 | 8 | 0 | 0 | node2vec_ppi +0.037 [-0.0438, 0.1177], 0.273, 0.188 |
| pd:per_head | II | all-16, PCA | 0.893 | 5 | 0 | 0 | node2vec_ppi +0.026 [-0.0217, 0.0729], 0.207, 0.250 |
| pd:per_head | III | selected combo, no PCA | 0.824 | 2 | 0 | 0 | bioconceptvec +0.026 [-0.0189, 0.0718], 0.181, 0.250 |
| pd:per_head | III | all-16, no PCA | 0.854 | 0 | 0 | 0 | bioconceptvec -0.003 [-0.0448, 0.0381], 0.832, 0.688 |
| pd:per_head | III | selected combo, PCA | 0.677 | 12 | 0 | 0 | bioconceptvec +0.173 [-0.0467, 0.3926], 0.094, 0.062 |
| pd:per_head | III | all-16, PCA | 0.723 | 10 | 0 | 0 | bioconceptvec +0.127 [-0.0264, 0.2808], 0.083, 0.062 |

### T5. A single embedding picked WITHOUT the fold's test set, minus the no-PCA multimodal model (fold-paired AUPRG)

sel-best: best selection-split score (the split the combo's selection used); val-best: best score on the fold's validation split (tier-matched); LOSO-best: best mean test AUPRG over the other 4 splits; oracle: best mean test AUPRG over all 5 splits (uses the test set: optimistic).

| Variant | Tier | Picker | Picks (folds 0-4) | Picked AUPRG | vs | Multimodal | Diff [95% CI] | p_t | p sign-flip |
|---|---|---|---|---|---|---|---|---|---|
| pd:pooled | II | sel-best | text_embed, scgpt, geneformer, text_embed, geneformer | 0.862 | combo | 0.943 | -0.081 [-0.184, +0.022] | 0.096 | 0.0625 |
| pd:pooled | II | sel-best | text_embed, scgpt, geneformer, text_embed, geneformer | 0.862 | allmod | 0.944 | -0.083 [-0.187, +0.022] | 0.094 | 0.0625 |
| pd:pooled | II | val-best | go2vec, text_embed, go2vec, text_embed, ppi_svd | 0.940 | combo | 0.943 | -0.002 [-0.020, +0.015] | 0.742 | 0.7500 |
| pd:pooled | II | val-best | go2vec, text_embed, go2vec, text_embed, ppi_svd | 0.940 | allmod | 0.944 | -0.004 [-0.016, +0.008] | 0.406 | 0.3125 |
| pd:pooled | II | LOSO-best | text_embed, text_embed, text_embed, text_embed, text_embed | 0.946 | combo | 0.943 | +0.004 [-0.022, +0.029] | 0.711 | 0.9375 |
| pd:pooled | II | LOSO-best | text_embed, text_embed, text_embed, text_embed, text_embed | 0.946 | allmod | 0.944 | +0.002 [-0.013, +0.017] | 0.744 | 0.8125 |
| pd:pooled | II | oracle-best(test,biased) | text_embed, text_embed, text_embed, text_embed, text_embed | 0.946 | combo | 0.943 | +0.004 [-0.022, +0.029] | 0.711 | 0.9375 |
| pd:pooled | II | oracle-best(test,biased) | text_embed, text_embed, text_embed, text_embed, text_embed | 0.946 | allmod | 0.944 | +0.002 [-0.013, +0.017] | 0.744 | 0.8125 |
| pd:pooled | III | sel-best | prot_t5, prot_t5, mashup, kg_complex, kg_complex | 0.675 | combo | 0.867 | -0.193 [-0.451, +0.066] | 0.108 | 0.0625 |
| pd:pooled | III | sel-best | prot_t5, prot_t5, mashup, kg_complex, kg_complex | 0.675 | allmod | 0.829 | -0.154 [-0.478, +0.170] | 0.256 | 0.3750 |
| pd:pooled | III | val-best | node2vec_ppi, node2vec_ppi, go, text_embed, mashup | 0.879 | combo | 0.867 | +0.012 [-0.010, +0.034] | 0.202 | 0.3750 |
| pd:pooled | III | val-best | node2vec_ppi, node2vec_ppi, go, text_embed, mashup | 0.879 | allmod | 0.829 | +0.050 [-0.040, +0.141] | 0.199 | 0.1250 |
| pd:pooled | III | LOSO-best | go2vec, text_embed, node2vec_ppi, go2vec, go2vec | 0.850 | combo | 0.867 | -0.017 [-0.063, +0.029] | 0.360 | 0.3125 |
| pd:pooled | III | LOSO-best | go2vec, text_embed, node2vec_ppi, go2vec, go2vec | 0.850 | allmod | 0.829 | +0.021 [-0.074, +0.116] | 0.570 | 0.5625 |
| pd:pooled | III | oracle-best(test,biased) | go2vec, go2vec, go2vec, go2vec, go2vec | 0.883 | combo | 0.867 | +0.015 [-0.058, +0.089] | 0.593 | 0.6250 |
| pd:pooled | III | oracle-best(test,biased) | go2vec, go2vec, go2vec, go2vec, go2vec | 0.883 | allmod | 0.829 | +0.054 [-0.101, +0.208] | 0.389 | 0.5000 |
| pd:per_head | II | sel-best | text_embed, kg_complex, geneformer, text_embed, prot_t5 | 0.864 | combo | 0.900 | -0.036 [-0.095, +0.023] | 0.163 | 0.1875 |
| pd:per_head | II | sel-best | text_embed, kg_complex, geneformer, text_embed, prot_t5 | 0.864 | allmod | 0.925 | -0.061 [-0.157, +0.036] | 0.155 | 0.0625 |
| pd:per_head | II | val-best | go, kg_complex, mashup, prot_t5, ppi_svd | 0.898 | combo | 0.900 | -0.003 [-0.058, +0.053] | 0.893 | 0.8750 |
| pd:per_head | II | val-best | go, kg_complex, mashup, prot_t5, ppi_svd | 0.898 | allmod | 0.925 | -0.027 [-0.063, +0.008] | 0.097 | 0.1250 |
| pd:per_head | II | LOSO-best | node2vec_ppi, node2vec_ppi, prot_t5, node2vec_ppi, node2vec_ppi | 0.904 | combo | 0.900 | +0.003 [-0.019, +0.026] | 0.722 | 0.6250 |
| pd:per_head | II | LOSO-best | node2vec_ppi, node2vec_ppi, prot_t5, node2vec_ppi, node2vec_ppi | 0.904 | allmod | 0.925 | -0.021 [-0.050, +0.007] | 0.105 | 0.0625 |
| pd:per_head | II | oracle-best(test,biased) | node2vec_ppi, node2vec_ppi, node2vec_ppi, node2vec_ppi, node2vec_ppi | 0.919 | combo | 0.900 | +0.018 [-0.042, +0.079] | 0.448 | 0.6250 |
| pd:per_head | II | oracle-best(test,biased) | node2vec_ppi, node2vec_ppi, node2vec_ppi, node2vec_ppi, node2vec_ppi | 0.919 | allmod | 0.925 | -0.006 [-0.026, +0.014] | 0.432 | 0.4375 |
| pd:per_head | III | sel-best | kg_complex, ppi_raw, kg_complex, esm1b, ppi_raw | 0.516 | combo | 0.824 | -0.308 [-0.801, +0.185] | 0.158 | 0.1250 |
| pd:per_head | III | sel-best | kg_complex, ppi_raw, kg_complex, esm1b, ppi_raw | 0.516 | allmod | 0.854 | -0.338 [-0.833, +0.157] | 0.131 | 0.0625 |
| pd:per_head | III | val-best | node2vec_ppi, node2vec_ppi, go, prot_t5, mashup | 0.843 | combo | 0.824 | +0.019 [-0.105, +0.143] | 0.695 | 0.5625 |
| pd:per_head | III | val-best | node2vec_ppi, node2vec_ppi, go, prot_t5, mashup | 0.843 | allmod | 0.854 | -0.011 [-0.123, +0.101] | 0.800 | 0.6250 |
| pd:per_head | III | LOSO-best | bioconceptvec, bioconceptvec, node2vec_ppi, bioconceptvec, bioconceptvec | 0.819 | combo | 0.824 | -0.004 [-0.096, +0.088] | 0.903 | 0.8125 |
| pd:per_head | III | LOSO-best | bioconceptvec, bioconceptvec, node2vec_ppi, bioconceptvec, bioconceptvec | 0.819 | allmod | 0.854 | -0.034 [-0.121, +0.052] | 0.334 | 0.3125 |
| pd:per_head | III | oracle-best(test,biased) | bioconceptvec, bioconceptvec, bioconceptvec, bioconceptvec, bioconceptvec | 0.850 | combo | 0.824 | +0.026 [-0.019, +0.072] | 0.181 | 0.2500 |
| pd:per_head | III | oracle-best(test,biased) | bioconceptvec, bioconceptvec, bioconceptvec, bioconceptvec, bioconceptvec | 0.850 | allmod | 0.854 | -0.003 [-0.045, +0.038] | 0.832 | 0.6875 |

### T6. Leave-one-category-out: ablated minus full combo (self-scored AUPRG, fold-paired; Holm over the 6 categories per arm x variant x tier)

Negative = the category helps. Source: `paper_tables/tab_loco.csv` (delta_vs_reference = delta_paired_mean, reproduced to 3e-16) and the `ablate_no_<cat>_<cv>` / `var_pd_pooled__ablate_no_<cat>_<cv>` results.json.

Arm none:

| Variant | Category removed | Tier I diff (p_t) | Tier II diff (p_t) | Tier III diff [95% CI] | III p_t / Holm / sign-flip | III # worse |
|---|---|---|---|---|---|---|
| pd:pooled | expression | -0.0003 (0.114) | -0.001 (0.671) | -0.015 [-0.052, +0.022] | 0.318 / 1.000 / 0.4375 | 3/5 |
| pd:pooled | go | -0.0002 (0.144) | +0.002 (0.842) | -0.010 [-0.038, +0.018] | 0.383 / 1.000 / 0.5625 | 3/5 |
| pd:pooled | kg | -0.0002 (0.184) | -0.001 (0.626) | -0.001 [-0.020, +0.017] | 0.870 / 1.000 / 1.0000 | 1/5 |
| pd:pooled | ppi | -0.0002 (0.178) | -0.001 (0.858) | -0.007 [-0.045, +0.030] | 0.615 / 1.000 / 0.6875 | 3/5 |
| pd:pooled | protein_seq | +0.0000 (0.765) | -0.008 (0.132) | -0.037 [-0.087, +0.013] | 0.107 / 0.641 / 0.0625 | 5/5 |
| pd:pooled | text | -0.0004 (0.010) | -0.020 (0.068) | -0.018 [-0.065, +0.029] | 0.355 / 1.000 / 0.4375 | 4/5 |
| pd:pooled | none (full combo) | 0.9897 | 0.943 | 0.867 |  |  |
| pd:per_head | expression | -0.0005 (0.148) | +0.004 (0.646) | -0.021 [-0.059, +0.018] | 0.211 / 0.845 / 0.2500 | 4/5 |
| pd:per_head | go | -0.0005 (0.652) | +0.002 (0.847) | -0.022 [-0.070, +0.025] | 0.260 / 0.845 / 0.2500 | 4/5 |
| pd:per_head | kg | -0.0013 (0.364) | -0.021 (0.258) | +0.013 [-0.030, +0.057] | 0.434 / 0.845 / 0.5000 | 2/5 |
| pd:per_head | ppi | -0.0011 (0.231) | -0.011 (0.079) | +0.019 [-0.019, +0.057] | 0.235 / 0.845 / 0.3125 | 1/5 |
| pd:per_head | protein_seq | -0.0017 (0.542) | -0.022 (0.210) | -0.025 [-0.064, +0.013] | 0.143 / 0.716 / 0.1875 | 4/5 |
| pd:per_head | text | -0.0036 (0.048) | -0.012 (0.667) | -0.047 [-0.104, +0.011] | 0.087 / 0.520 / 0.1250 | 4/5 |
| pd:per_head | none (full combo) | 0.9748 | 0.900 | 0.824 |  |  |

Arm joint:

| Variant | Category removed | Tier I diff (p_t) | Tier II diff (p_t) | Tier III diff [95% CI] | III p_t / Holm / sign-flip | III # worse |
|---|---|---|---|---|---|---|
| pd:pooled | expression | -0.0001 (0.507) | -0.011 (0.496) | +0.028 [-0.020, +0.077] | 0.182 / 1.000 / 0.2500 | 1/5 |
| pd:pooled | go | +0.0000 (0.803) | -0.004 (0.244) | +0.017 [-0.059, +0.094] | 0.568 / 1.000 / 0.4375 | 1/5 |
| pd:pooled | kg | -0.0001 (0.277) | -0.003 (0.332) | +0.012 [-0.055, +0.079] | 0.655 / 1.000 / 0.6875 | 2/5 |
| pd:pooled | ppi | +0.0001 (0.557) | -0.004 (0.452) | +0.002 [-0.079, +0.083] | 0.951 / 1.000 / 0.8750 | 3/5 |
| pd:pooled | protein_seq | -0.0001 (0.365) | -0.008 (0.157) | -0.045 [-0.213, +0.124] | 0.502 / 1.000 / 0.6250 | 3/5 |
| pd:pooled | text | -0.0001 (0.402) | -0.014 (0.128) | -0.018 [-0.157, +0.120] | 0.730 / 1.000 / 0.8125 | 3/5 |
| pd:pooled | none (full combo) | 0.9887 | 0.937 | 0.818 |  |  |
| pd:per_head | expression | -0.0011 (0.655) | -0.001 (0.872) | +0.007 [-0.234, +0.248] | 0.939 / 1.000 / 0.9375 | 2/5 |
| pd:per_head | go | +0.0008 (0.601) | +0.000 (0.953) | +0.058 [-0.110, +0.226] | 0.393 / 1.000 / 0.5000 | 1/5 |
| pd:per_head | kg | -0.0034 (0.126) | -0.008 (0.614) | +0.084 [-0.049, +0.216] | 0.154 / 0.926 / 0.1250 | 1/5 |
| pd:per_head | ppi | -0.0002 (0.923) | -0.024 (0.154) | +0.087 [-0.067, +0.241] | 0.192 / 0.959 / 0.3125 | 1/5 |
| pd:per_head | protein_seq | -0.0004 (0.856) | +0.001 (0.888) | -0.025 [-0.174, +0.123] | 0.660 / 1.000 / 0.6250 | 4/5 |
| pd:per_head | text | +0.0004 (0.780) | -0.005 (0.321) | -0.003 [-0.061, +0.055] | 0.903 / 1.000 / 0.8125 | 3/5 |
| pd:per_head | none (full combo) | 0.9737 | 0.882 | 0.677 |  |  |

### T7. Modalities picked per tier and split (identical in both arms)

| Tier | Category | split 0 | split 1 | split 2 | split 3 | split 4 |
|---|---|---|---|---|---|---|
| I | expression | geneformer | scgpt | scgpt | geneformer | scgpt |
| I | go | go2vec | go2vec | go | go2vec | go2vec |
| I | kg | kg_complex | kg_complex | kg_complex | kg_complex | kg_complex |
| I | ppi | ppi_raw | ppi_raw | ppi_raw | ppi_raw | ppi_raw |
| I | protein_seq | prot_t5 | esm1b | esm2 | prot_t5 | esm1b |
| I | text | text_embed | text_embed | text_embed | text_embed | text_embed |
| II | expression | geneformer | scgpt | geneformer | geneformer | geneformer |
| II | go | go | go | go | go2vec | go |
| II | kg | kg_complex | kg_complex | kg_complex | kg_complex | kg_complex |
| II | ppi | mashup | node2vec_ppi | ppi_raw | node2vec_ppi | ppi_raw |
| II | protein_seq | esm1b | esm1b | prot_t5 | prot_t5 | esm1b |
| II | text | text_embed | text_embed | text_embed | text_embed | text_embed |
| III | expression | scgpt | geneformer | geneformer | geneformer | geneformer |
| III | go | go2vec | go | go | go2vec | go2vec |
| III | kg | kg_complex | kg_complex | kg_complex | kg_complex | kg_complex |
| III | ppi | mashup | ppi_raw | mashup | ppi_raw | ppi_svd |
| III | protein_seq | prot_t5 | prot_t5 | esm1b | esm1b | esm1b |
| III | text | text_embed | text_embed | text_embed | text_embed | text_embed |

### T8. Does the selection score predict test performance? Spearman across the 16 single-embedding runs, per split

| Variant | Tier | selection split vs test: mean (min, max) | val split vs test: mean (min, max) | spread of selection scores | spread of test AUPRG |
|---|---|---|---|---|---|
| pd:per_head | I | 0.94 (0.91, 0.96) | 0.93 (0.89, 0.97) | 0.033 | 0.033 |
| pd:per_head | II | 0.29 (0.01, 0.48) | 0.68 (0.48, 0.83) | 0.040 | 0.155 |
| pd:per_head | III | 0.14 (-0.04, 0.34) | 0.42 (0.02, 0.66) | 0.041 | 0.643 |
| pd:pooled | I | 0.96 (0.92, 0.98) | 0.95 (0.91, 0.99) | 0.015 | 0.014 |
| pd:pooled | II | 0.10 (-0.31, 0.41) | 0.61 (0.13, 0.81) | 0.017 | 0.150 |
| pd:pooled | III | -0.11 (-0.33, 0.18) | 0.61 (0.28, 0.85) | 0.017 | 0.784 |


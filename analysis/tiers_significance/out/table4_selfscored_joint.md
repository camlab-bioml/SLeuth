| Tier | A − B | Metric | mean A | mean B | Δ | split-level 95% t-CI of Δ | folds A>B | paired t p (Holm) | sign-flip p | boot CI of Δ | p_boot (Holm) |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| I | cell_emb=0 +bias=0 − as-trained | AUROC | 0.917 | 0.918 | −0.001 | [−0.003, 0.001] | 1/5 | 0.173 (0.173) | 0.188 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=mean − as-trained | AUROC | 0.780 | 0.918 | −0.139 | [−0.229, −0.049] | 0/5 | 0.013 (0.026) | 0.0625 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=0 − as-trained | AUROC | 0.805 | 0.918 | −0.113 | [−0.164, −0.063] | 0/5 | 0.003 (0.010) | 0.0625 | no bootstrap pair | n/a |
| I | cell_emb=0 +bias=0 − as-trained | AUPR (AP) | 0.825 | 0.827 | −0.003 | [−0.007, 0.001] | 1/5 | 0.144 (0.144) | 0.125 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=mean − as-trained | AUPR (AP) | 0.610 | 0.827 | −0.217 | [−0.294, −0.140] | 0/5 | 0.001 (0.004) | 0.0625 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=0 − as-trained | AUPR (AP) | 0.667 | 0.827 | −0.160 | [−0.241, −0.079] | 0/5 | 0.005 (0.011) | 0.0625 | no bootstrap pair | n/a |
| I | cell_emb=0 +bias=0 − as-trained | AUPRG | 0.973 | 0.974 | −0.001 | [−0.003, 0.001] | 1/5 | 0.138 (0.138) | 0.125 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=mean − as-trained | AUPRG | 0.883 | 0.974 | −0.091 | [−0.151, −0.031] | 0/5 | 0.014 (0.028) | 0.0625 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=0 − as-trained | AUPRG | 0.915 | 0.974 | −0.058 | [−0.091, −0.026] | 0/5 | 0.008 (0.023) | 0.0625 | no bootstrap pair | n/a |
| II | cell_emb=0 +bias=0 − as-trained | AUROC | 0.794 | 0.789 | 0.005 | [−0.003, 0.014] | 4/5 | 0.164 (0.491) | 0.125 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=mean − as-trained | AUROC | 0.799 | 0.789 | 0.010 | [−0.054, 0.075] | 3/5 | 0.682 (1.000) | 0.688 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=0 − as-trained | AUROC | 0.789 | 0.789 | 0.001 | [−0.087, 0.089] | 2/5 | 0.978 (1.000) | 1.000 | no bootstrap pair | n/a |
| II | cell_emb=0 +bias=0 − as-trained | AUPR (AP) | 0.625 | 0.622 | 0.003 | [−0.002, 0.009] | 4/5 | 0.190 (0.569) | 0.250 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=mean − as-trained | AUPR (AP) | 0.574 | 0.622 | −0.048 | [−0.141, 0.045] | 1/5 | 0.228 (0.569) | 0.312 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=0 − as-trained | AUPR (AP) | 0.575 | 0.622 | −0.047 | [−0.164, 0.069] | 2/5 | 0.322 (0.569) | 0.375 | no bootstrap pair | n/a |
| II | cell_emb=0 +bias=0 − as-trained | AUPRG | 0.884 | 0.882 | 0.002 | [−0.003, 0.008] | 3/5 | 0.310 (0.931) | 0.375 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=mean − as-trained | AUPRG | 0.867 | 0.882 | −0.015 | [−0.075, 0.045] | 1/5 | 0.532 (1.000) | 0.500 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=0 − as-trained | AUPRG | 0.868 | 0.882 | −0.014 | [−0.095, 0.067] | 2/5 | 0.660 (1.000) | 0.625 | no bootstrap pair | n/a |
| III | cell_emb=0 +bias=0 − as-trained | AUROC | 0.668 | 0.665 | 0.003 | [−0.013, 0.018] | 2/5 | 0.671 (1.000) | 0.750 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=mean − as-trained | AUROC | 0.667 | 0.665 | 0.002 | [−0.034, 0.038] | 3/5 | 0.911 (1.000) | 0.938 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=0 − as-trained | AUROC | 0.689 | 0.665 | 0.024 | [−0.013, 0.061] | 3/5 | 0.148 (0.444) | 0.250 | no bootstrap pair | n/a |
| III | cell_emb=0 +bias=0 − as-trained | AUPR (AP) | 0.423 | 0.419 | 0.004 | [−0.010, 0.018] | 3/5 | 0.459 (0.917) | 0.438 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=mean − as-trained | AUPR (AP) | 0.413 | 0.419 | −0.006 | [−0.046, 0.033] | 2/5 | 0.671 (0.917) | 0.688 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=0 − as-trained | AUPR (AP) | 0.437 | 0.419 | 0.018 | [−0.016, 0.052] | 3/5 | 0.217 (0.651) | 0.250 | no bootstrap pair | n/a |
| III | cell_emb=0 +bias=0 − as-trained | AUPRG | 0.692 | 0.677 | 0.014 | [−0.021, 0.050] | 3/5 | 0.321 (0.654) | 0.438 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=mean − as-trained | AUPRG | 0.669 | 0.677 | −0.008 | [−0.084, 0.067] | 2/5 | 0.773 (0.773) | 0.688 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=0 − as-trained | AUPRG | 0.723 | 0.677 | 0.045 | [−0.041, 0.132] | 3/5 | 0.218 (0.654) | 0.375 | no bootstrap pair | n/a |

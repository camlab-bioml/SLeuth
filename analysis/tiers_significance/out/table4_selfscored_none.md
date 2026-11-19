| Tier | A − B | Metric | mean A | mean B | Δ | split-level 95% t-CI of Δ | folds A>B | paired t p (Holm) | sign-flip p | boot CI of Δ | p_boot (Holm) |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| I | cell_emb=0 +bias=0 − as-trained | AUROC | 0.918 | 0.918 | 0.000 | [−0.002, 0.002] | 3/5 | 0.731 (1.000) | 0.750 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=mean − as-trained | AUROC | 0.915 | 0.918 | −0.003 | [−0.018, 0.012] | 2/5 | 0.627 (1.000) | 0.625 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=0 − as-trained | AUROC | 0.924 | 0.918 | 0.006 | [−0.008, 0.020] | 3/5 | 0.273 (0.818) | 0.438 | no bootstrap pair | n/a |
| I | cell_emb=0 +bias=0 − as-trained | AUPR (AP) | 0.828 | 0.829 | −0.001 | [−0.007, 0.004] | 4/5 | 0.597 (1.000) | 1.000 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=mean − as-trained | AUPR (AP) | 0.794 | 0.829 | −0.035 | [−0.081, 0.010] | 1/5 | 0.096 (0.288) | 0.188 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=0 − as-trained | AUPR (AP) | 0.826 | 0.829 | −0.004 | [−0.026, 0.019] | 2/5 | 0.681 (1.000) | 0.688 | no bootstrap pair | n/a |
| I | cell_emb=0 +bias=0 − as-trained | AUPRG | 0.975 | 0.975 | 0.000 | [−0.001, 0.001] | 4/5 | 0.602 (1.000) | 1.000 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=mean − as-trained | AUPRG | 0.968 | 0.975 | −0.007 | [−0.018, 0.004] | 2/5 | 0.144 (0.432) | 0.250 | no bootstrap pair | n/a |
| I | cell_emb=0 +head_proj_bias=0 − as-trained | AUPRG | 0.975 | 0.975 | 0.001 | [−0.005, 0.006] | 2/5 | 0.770 (1.000) | 0.750 | no bootstrap pair | n/a |
| II | cell_emb=0 +bias=0 − as-trained | AUROC | 0.823 | 0.820 | 0.003 | [−0.003, 0.009] | 3/5 | 0.252 (0.707) | 0.375 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=mean − as-trained | AUROC | 0.806 | 0.820 | −0.014 | [−0.042, 0.014] | 1/5 | 0.236 (0.707) | 0.312 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=0 − as-trained | AUROC | 0.802 | 0.820 | −0.017 | [−0.084, 0.050] | 3/5 | 0.512 (0.707) | 0.750 | no bootstrap pair | n/a |
| II | cell_emb=0 +bias=0 − as-trained | AUPR (AP) | 0.655 | 0.650 | 0.004 | [−0.003, 0.012] | 3/5 | 0.190 (0.379) | 0.250 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=mean − as-trained | AUPR (AP) | 0.595 | 0.650 | −0.055 | [−0.126, 0.016] | 1/5 | 0.097 (0.291) | 0.125 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=0 − as-trained | AUPR (AP) | 0.604 | 0.650 | −0.047 | [−0.161, 0.068] | 2/5 | 0.320 (0.379) | 0.375 | no bootstrap pair | n/a |
| II | cell_emb=0 +bias=0 − as-trained | AUPRG | 0.905 | 0.900 | 0.004 | [−0.005, 0.013] | 3/5 | 0.276 (0.615) | 0.250 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=mean − as-trained | AUPRG | 0.878 | 0.900 | −0.023 | [−0.065, 0.019] | 1/5 | 0.205 (0.615) | 0.188 | no bootstrap pair | n/a |
| II | cell_emb=0 +head_proj_bias=0 − as-trained | AUPRG | 0.881 | 0.900 | −0.019 | [−0.095, 0.057] | 2/5 | 0.524 (0.615) | 0.438 | no bootstrap pair | n/a |
| III | cell_emb=0 +bias=0 − as-trained | AUROC | 0.744 | 0.735 | 0.009 | [−0.009, 0.027] | 3/5 | 0.230 (0.461) | 0.250 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=mean − as-trained | AUROC | 0.732 | 0.735 | −0.002 | [−0.053, 0.048] | 2/5 | 0.902 (0.902) | 0.938 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=0 − as-trained | AUROC | 0.760 | 0.735 | 0.026 | [−0.011, 0.062] | 4/5 | 0.125 (0.376) | 0.188 | no bootstrap pair | n/a |
| III | cell_emb=0 +bias=0 − as-trained | AUPR (AP) | 0.536 | 0.523 | 0.013 | [−0.010, 0.036] | 3/5 | 0.195 (0.307) | 0.250 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=mean − as-trained | AUPR (AP) | 0.493 | 0.523 | −0.030 | [−0.076, 0.017] | 1/5 | 0.153 (0.307) | 0.188 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=0 − as-trained | AUPR (AP) | 0.549 | 0.523 | 0.026 | [−0.002, 0.054] | 4/5 | 0.061 (0.184) | 0.125 | no bootstrap pair | n/a |
| III | cell_emb=0 +bias=0 − as-trained | AUPRG | 0.843 | 0.824 | 0.019 | [−0.015, 0.053] | 3/5 | 0.193 (0.387) | 0.312 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=mean − as-trained | AUPRG | 0.805 | 0.824 | −0.019 | [−0.066, 0.029] | 1/5 | 0.341 (0.387) | 0.438 | no bootstrap pair | n/a |
| III | cell_emb=0 +head_proj_bias=0 − as-trained | AUPRG | 0.859 | 0.824 | 0.036 | [−0.010, 0.081] | 4/5 | 0.096 (0.287) | 0.188 | no bootstrap pair | n/a |

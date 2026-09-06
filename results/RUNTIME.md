# Measured runtime

System: gpu NVIDIA GeForce RTX 5070, torch 2.11.0+cu128, cuda 12.8

Per-configuration compute time (seconds, five folds; image caching is included only when the cache was built during that run):

| Configuration | Seconds |
|---|---:|
| convae_CGI_leg | 163.2 |
| convae_FDEI_leg | 132.3 |
| convae_GEI_leg | 132.4 |
| convae_HEI_leg | 156.8 |
| patchcore_CGI_full | 74.4 |
| patchcore_CGI_leg | 101.2 |
| patchcore_FDEI_full | 35.4 |
| patchcore_FDEI_leg | 70.7 |
| patchcore_GEI_CGI_FDEI_leg | 16.6 |
| patchcore_GEI_full | 60.9 |
| patchcore_GEI_leg | 72.4 |
| patchcore_HEI_comb1_leg | 85.6 |
| patchcore_HEI_comb2_leg | 90.5 |
| patchcore_HEI_comb3_leg | 89.1 |
| patchcore_HEI_comb4_leg | 88.3 |
| patchcore_HEI_comb5_leg | 84.7 |
| patchcore_HEI_comb6_leg | 88.2 |
| patchcore_HEI_full | 75.4 |
| patchcore_HEI_leg | 84.9 |
| stfpm_CGI_leg | 145.0 |
| stfpm_FDEI_leg | 121.3 |
| stfpm_GEI_leg | 121.0 |
| stfpm_HEI_leg | 141.3 |
| subspace_CGI_full | 3.8 |
| subspace_CGI_leg | 4.0 |
| subspace_FDEI_full | 1.8 |
| subspace_FDEI_leg | 1.8 |
| subspace_GEI_CGI_FDEI_leg | 3.8 |
| subspace_GEI_full | 1.8 |
| subspace_GEI_leg | 1.8 |
| subspace_HEI_full | 4.2 |
| subspace_HEI_leg | 3.9 |

Sum over configurations: 2258.6 s

| Stage | Seconds |
|---|---:|
| Main grid wall time (scripts/run_main.py) | 1457.1 |
| Design-factor analyses wall time, last invocation (scripts/run_ablations.py) | 13.0 |
| Energy-image cross-check (scripts/regenerate_ei.py) | 24.9 |
| Segmentation validation, val split | 21.6 |
| Segmentation validation, test split | 10.7 |

Completed configurations are reused on re-invocation, so a stage wall time can be far below the sum of its configurations.

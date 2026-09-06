# Re-execution versus manuscript values

Main grid status: COMPLETE

Values reported in the manuscript were obtained on a GTX 1080 8 GB / CUDA 11.8 / PyTorch 2.5.1 system. Re-execution with this repository on the system recorded in results/RUNTIME.md yields the values below; differences are tabulated and are consistent with library-version, GPU-numerics and random-sampling effects. Settings are fixed and were not tuned to manuscript outcomes. Rows labeled fold-mean are means of the five fold values; rows labeled pooled are computed on the pooled out-of-fold scores (the permutation and detected-count rows also use pooled scores).
The manuscript's Evaluation protocol identifies its reported point estimate as the metric on the pooled out-of-fold scores, with the 95% interval taken across the five folds. The pooled row is therefore the like-for-like comparison; the fold-mean row is compared against the same manuscript value so that both estimands are visible. Approximate fusion/per-cow references do not state the aggregation; both are shown for those too. NA means the manuscript does not report an exact point estimate. Rows marked (manuscript figure, read) compare against bar heights read from a figure (about +/-0.01). Permutation rows give k/B for k exceedances among B permutations (reported as <1/B when k = 0) followed by the (k+1)/(B+1) estimate.

## Analysis cohort

Aligned windows 10086, evaluable passes 252, known cows 155, GroupKFold groups 156; consensus counts score 1: 120, score 2: 126, score 3: 6; crowding counts 1: 27, 2: 60, 3: 165.

## Comparison table

| Metric | This repository | Manuscript | \|Δ\| |
|---|---:|---:|---:|
| convae_CGI_leg pooled AUROC (manuscript figure, read) | 0.545960 | 0.568000 | 0.022040 |
| convae_CGI_leg fold-mean AUROC (manuscript figure, read) | 0.576571 | 0.568000 | 0.008571 |
| convae_FDEI_leg pooled AUROC (manuscript figure, read) | 0.509280 | 0.520000 | 0.010720 |
| convae_FDEI_leg fold-mean AUROC (manuscript figure, read) | 0.521072 | 0.520000 | 0.001072 |
| convae_GEI_leg pooled AUROC (manuscript figure, read) | 0.562563 | 0.525000 | 0.037563 |
| convae_GEI_leg fold-mean AUROC (manuscript figure, read) | 0.568322 | 0.525000 | 0.043322 |
| convae_HEI_leg pooled AUROC (manuscript figure, read) | 0.476010 | 0.485000 | 0.008990 |
| convae_HEI_leg fold-mean AUROC (manuscript figure, read) | 0.512432 | 0.485000 | 0.027432 |
| patchcore_CGI_full pooled AUROC | 0.499306 | 0.502000 | 0.002694 |
| patchcore_CGI_full fold-mean AUROC | 0.501858 | 0.502000 | 0.000142 |
| patchcore_CGI_leg pooled AUROC | 0.671780 | 0.668000 | 0.003780 |
| patchcore_CGI_leg fold-mean AUROC | 0.678626 | 0.668000 | 0.010626 |
| patchcore crowding 1 pooled AUROC | 0.818182 | 0.807000 | 0.011182 |
| patchcore crowding 2 pooled AUROC | 0.535714 | 0.625000 | 0.089286 |
| patchcore crowding 3 pooled AUROC | 0.696796 | 0.661000 | 0.035796 |
| patchcore_CGI_leg permutation p, k/B; (k+1)/(B+1) | 0/10000 (<1/10000); (k+1)/(B+1) = 9.999e-05 | <0.001 | NA |
| patchcore_CGI_leg permutation mean AUROC | 0.500101 | 0.500000 | 0.000101 |
| patchcore_CGI_leg mild detected | 119 | 116 | 3 |
| patchcore_CGI_leg severe detected | 6 | 5 | 1 |
| patchcore_CGI_leg fold-mean auprc | 0.728928 | 0.690000 | 0.038928 |
| patchcore_CGI_leg pooled auprc | 0.714552 | 0.690000 | 0.024552 |
| patchcore_CGI_leg fold-mean max_f1 | 0.727138 | 0.697000 | 0.030138 |
| patchcore_CGI_leg pooled max_f1 | 0.696379 | 0.697000 | 0.000621 |
| patchcore_CGI_leg pooled Spearman rho | 0.304814 | 0.300000 | 0.004814 |
| patchcore_CGI_leg pooled median score, severity 1 | 6.878437 | 7.240000 | 0.361563 |
| patchcore_CGI_leg pooled median score, severity 2 | 7.161333 | 7.450000 | 0.288667 |
| patchcore_CGI_leg pooled median score, severity 3 | 7.545608 | 7.740000 | 0.194392 |
| PatchCore CGI AUROC CI low | 0.549416 | 0.580000 | 0.030584 |
| PatchCore CGI AUROC CI high | 0.807835 | 0.770000 | 0.037835 |
| patchcore_CGI_percow_leg pooled AUROC | 0.543434 | NA | NA |
| patchcore_CGI_percow_leg fold-mean AUROC | 0.539792 | NA | NA |
| patchcore_FDEI_full pooled AUROC | 0.505051 | 0.495000 | 0.010051 |
| patchcore_FDEI_full fold-mean AUROC | 0.521428 | 0.495000 | 0.026428 |
| patchcore_FDEI_leg pooled AUROC | 0.643750 | 0.596000 | 0.047750 |
| patchcore_FDEI_leg fold-mean AUROC | 0.650135 | 0.596000 | 0.054135 |
| patchcore_GEI_CGI_FDEI_leg pooled AUROC | 0.662437 | NA | NA |
| patchcore_GEI_CGI_FDEI_leg fold-mean AUROC | 0.665159 | NA | NA |
| patchcore_GEI_full pooled AUROC | 0.479167 | 0.473000 | 0.006167 |
| patchcore_GEI_full fold-mean AUROC | 0.488114 | 0.473000 | 0.015114 |
| patchcore_GEI_leg pooled AUROC | 0.624874 | 0.625000 | 0.000126 |
| patchcore_GEI_leg fold-mean AUROC | 0.634248 | 0.625000 | 0.009248 |
| patchcore_HEI_comb1_leg pooled AUROC | 0.537879 | NA | NA |
| patchcore_HEI_comb1_leg fold-mean AUROC | 0.539963 | NA | NA |
| patchcore_HEI_comb2_leg pooled AUROC | 0.521654 | NA | NA |
| patchcore_HEI_comb2_leg fold-mean AUROC | 0.523142 | NA | NA |
| patchcore_HEI_comb3_leg pooled AUROC | 0.527083 | NA | NA |
| patchcore_HEI_comb3_leg fold-mean AUROC | 0.529973 | NA | NA |
| patchcore_HEI_comb4_leg pooled AUROC | 0.529040 | NA | NA |
| patchcore_HEI_comb4_leg fold-mean AUROC | 0.533606 | NA | NA |
| patchcore_HEI_comb5_leg pooled AUROC | 0.540720 | NA | NA |
| patchcore_HEI_comb5_leg fold-mean AUROC | 0.545927 | NA | NA |
| patchcore_HEI_comb6_leg pooled AUROC | 0.531124 | NA | NA |
| patchcore_HEI_comb6_leg fold-mean AUROC | 0.531693 | NA | NA |
| patchcore_HEI_full pooled AUROC | 0.521843 | 0.507000 | 0.014843 |
| patchcore_HEI_full fold-mean AUROC | 0.521166 | 0.507000 | 0.014166 |
| patchcore_HEI_leg pooled AUROC | 0.531124 | 0.515000 | 0.016124 |
| patchcore_HEI_leg fold-mean AUROC | 0.531693 | 0.515000 | 0.016693 |
| stfpm_CGI_leg pooled AUROC (manuscript figure, read) | 0.590972 | 0.567000 | 0.023972 |
| stfpm_CGI_leg fold-mean AUROC (manuscript figure, read) | 0.591470 | 0.567000 | 0.024470 |
| stfpm_FDEI_leg pooled AUROC (manuscript figure, read) | 0.576452 | 0.530000 | 0.046452 |
| stfpm_FDEI_leg fold-mean AUROC (manuscript figure, read) | 0.583716 | 0.530000 | 0.053716 |
| stfpm_GEI_leg pooled AUROC (manuscript figure, read) | 0.575126 | 0.578000 | 0.002874 |
| stfpm_GEI_leg fold-mean AUROC (manuscript figure, read) | 0.578636 | 0.578000 | 0.000636 |
| stfpm_HEI_leg pooled AUROC (manuscript figure, read) | 0.570833 | 0.510000 | 0.060833 |
| stfpm_HEI_leg fold-mean AUROC (manuscript figure, read) | 0.567535 | 0.510000 | 0.057535 |
| subspace_CGI_full pooled AUROC | 0.602210 | 0.631000 | 0.028790 |
| subspace_CGI_full fold-mean AUROC | 0.605351 | 0.631000 | 0.025649 |
| subspace_CGI_leg pooled AUROC | 0.597917 | 0.635000 | 0.037083 |
| subspace_CGI_leg fold-mean AUROC | 0.599250 | 0.635000 | 0.035750 |
| subspace crowding 1 pooled AUROC | 0.676136 | 0.619000 | 0.057136 |
| subspace crowding 2 pooled AUROC | 0.482143 | 0.598000 | 0.115857 |
| subspace crowding 3 pooled AUROC | 0.633010 | 0.655000 | 0.021990 |
| subspace_CGI_leg permutation p, k/B; (k+1)/(B+1) | 37/10000; (k+1)/(B+1) = 0.0038 | <0.001 | NA |
| subspace_CGI_leg permutation mean AUROC | 0.500439 | 0.500000 | 0.000439 |
| subspace_CGI_leg mild detected | 125 | NA | NA |
| subspace_CGI_leg severe detected | 6 | 6 | 0 |
| subspace_FDEI_full pooled AUROC | 0.614331 | 0.620000 | 0.005669 |
| subspace_FDEI_full fold-mean AUROC | 0.614184 | 0.620000 | 0.005816 |
| subspace_FDEI_leg pooled AUROC | 0.608902 | 0.599000 | 0.009902 |
| subspace_FDEI_leg fold-mean AUROC | 0.616096 | 0.599000 | 0.017096 |
| subspace_GEI_CGI_FDEI_leg pooled AUROC | 0.605429 | NA | NA |
| subspace_GEI_CGI_FDEI_leg fold-mean AUROC | 0.605298 | NA | NA |
| subspace_GEI_full pooled AUROC | 0.602399 | 0.614000 | 0.011601 |
| subspace_GEI_full fold-mean AUROC | 0.605986 | 0.614000 | 0.008014 |
| subspace_GEI_leg pooled AUROC | 0.628788 | 0.625000 | 0.003788 |
| subspace_GEI_leg fold-mean AUROC | 0.629040 | 0.625000 | 0.004040 |
| subspace_HEI_full pooled AUROC | 0.538131 | 0.532000 | 0.006131 |
| subspace_HEI_full fold-mean AUROC | 0.531686 | 0.532000 | 0.000314 |
| subspace_HEI_leg pooled AUROC | 0.534596 | 0.535000 | 0.000404 |
| subspace_HEI_leg fold-mean AUROC | 0.533698 | 0.535000 | 0.001302 |
| Subspace fusion fold-mean AUROC | 0.605298 | 0.580000 | 0.025298 |
| Subspace fusion pooled AUROC | 0.605429 | 0.580000 | 0.025429 |
| PatchCore fusion fold-mean AUROC | 0.665159 | NA | NA |
| PatchCore fusion pooled AUROC | 0.662437 | NA | NA |
| Per-cow normalization fold-mean AUROC | 0.539792 | 0.550000 | 0.010208 |
| Per-cow normalization pooled AUROC | 0.543434 | 0.550000 | 0.006566 |
| HEI permutation minimum pooled AUROC | 0.521654 | 0.500000 | 0.021654 |
| HEI permutation maximum pooled AUROC | 0.540720 | 0.520000 | 0.020720 |
| HEI permutation minimum fold-mean AUROC | 0.523142 | 0.500000 | 0.023142 |
| HEI permutation maximum fold-mean AUROC | 0.545927 | 0.520000 | 0.025927 |
| Evaluable passes | 252 | 251 | 1 |

## Table 2: full image versus leg-region crop (fold-mean AUROC)

| Detector | Modality | Full (this repository) | Full (manuscript) | Leg (this repository) | Leg (manuscript) | Leg-full (this repository) | Leg-full (manuscript) |
|---|---|---:|---:|---:|---:|---:|---:|
| patchcore | CGI | 0.502 | 0.502 | 0.679 | 0.668 | +0.177 | +0.166 |
| patchcore | GEI | 0.488 | 0.473 | 0.634 | 0.625 | +0.146 | +0.152 |
| patchcore | FDEI | 0.521 | 0.495 | 0.650 | 0.596 | +0.129 | +0.101 |
| patchcore | HEI | 0.521 | 0.507 | 0.532 | 0.515 | +0.011 | +0.008 |
| subspace | CGI | 0.605 | 0.631 | 0.599 | 0.635 | -0.006 | +0.004 |
| subspace | GEI | 0.606 | 0.614 | 0.629 | 0.625 | +0.023 | +0.011 |
| subspace | FDEI | 0.614 | 0.620 | 0.616 | 0.599 | +0.002 | -0.021 |
| subspace | HEI | 0.532 | 0.532 | 0.534 | 0.535 | +0.002 | +0.003 |

## HEI channel permutations (PatchCore, leg crop); manuscript band 0.50-0.52

| Permutation | Fold-mean AUROC | 95% CI | Pooled AUROC |
|---:|---:|---|---:|
| 1 | 0.540 | [0.477, 0.603] | 0.538 |
| 2 | 0.523 | [0.438, 0.608] | 0.522 |
| 3 | 0.530 | [0.458, 0.602] | 0.527 |
| 4 | 0.534 | [0.448, 0.619] | 0.529 |
| 5 | 0.546 | [0.468, 0.624] | 0.541 |
| 6 | 0.532 | [0.467, 0.597] | 0.531 |

## Late fusion (GEI + CGI + FDEI, leg crop) and per-cow normalization

Manuscript (Results, design-factor analysis; stated as approximate): Subspace late fusion about 0.58 versus about 0.64 for the single best modality (Subspace CGI, 0.635 in manuscript Table 2); per-cow normalization about 0.55. The manuscript reports no PatchCore late-fusion value, so the PatchCore fusion rows of the comparison table have no manuscript counterpart.

| Analysis | Fold-mean AUROC | 95% CI | Pooled AUROC |
|---|---:|---|---:|
| Subspace, concatenated pass-mean vectors | 0.605 | [0.479, 0.731] | 0.605 |
| PatchCore, concatenated patch features | 0.665 | [0.532, 0.798] | 0.662 |
| PatchCore-CGI-leg, per-cow z-score | 0.540 | [0.389, 0.691] | 0.543 |

## AUROC discrepancies greater than 0.03

Listed on the pooled estimand, the one the manuscript reports; a fold-mean row is listed only where the same quantity has no pooled counterpart in the table above.

- convae_GEI_leg pooled AUROC (manuscript figure, read): 0.037563
- patchcore crowding 2 pooled AUROC: 0.089286
- patchcore crowding 3 pooled AUROC: 0.035796
- patchcore_FDEI_leg pooled AUROC: 0.047750
- stfpm_FDEI_leg pooled AUROC (manuscript figure, read): 0.046452
- stfpm_HEI_leg pooled AUROC (manuscript figure, read): 0.060833
- subspace_CGI_leg pooled AUROC: 0.037083
- subspace crowding 1 pooled AUROC: 0.057136
- subspace crowding 2 pooled AUROC: 0.115857

## Permutation p values not below the manuscript's <0.001

- subspace_CGI_leg: 37/10000; (k+1)/(B+1) = 0.0038, whereas the manuscript states <0.001.

## Interpretation limits

- OOF scores from separately fitted folds are pooled on their raw scale, as described in the Methods. Cross-fold score calibration is not established.
- Maximum F1 and severity thresholds are evaluation-set oracle quantities; no deployable threshold is claimed.
- The permutation test exchanges pass labels on fixed OOF scores. It neither retrains the models nor accounts for within-cow dependence or selecting a configuration from the grid.
- Per-cow normalization uses other evaluation scores, is transductive, and mixes standardized multi-pass cows with raw singleton scores.
- Held-out AUROC curves are diagnostic only: all learned runs use epoch 40, with no checkpoint selection.
- See docs/IMPLEMENTATION_NOTES.md, docs/EI_XCHECK.md and docs/SEG_XCHECK.md for implementation choices and upstream checks.

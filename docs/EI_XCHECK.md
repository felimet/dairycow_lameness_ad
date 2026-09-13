# Energy-image cross-check

Recording 001, pass 01; 276 masks; 38 retained windows (`keep_windows_w30`).

| Modality | Mean window MAE [0,1] | Mean window Pearson r | Archived mean / max | Regenerated mean / max |
|---|---:|---:|---:|---:|
| CGI | 0.151040 | 0.786197 | 0.469 / 1.000 | 0.499 / 1.000 |
| FDEI | 0.219684 | 0.870817 | 0.204 / 0.521 | 0.393 / 1.000 |
| GEI | 0.109910 | 0.863180 | 0.469 / 1.000 | 0.499 / 1.000 |
| HEI_comb6 | 0.076317 | 0.896377 | 0.225 / 1.000 | 0.240 / 1.000 |

Command: `conda run -n cowlame python scripts/regenerate_ei.py --data-root F:/cow-data/01_data --out results --recording 1 --pass-id 1 --window-step 5 --front-direction auto --window-list w30 --compare-list gated`.

Window size 41 frames, taken from mask_meta.json. Metadata omit the source window hop. The exposed 5-frame hop and one-based window origin are implementation assumptions; the mask metadata do not record them.
Window list `keep_windows_w30` of mask_meta.json: its 38 retained metadata indices 5..42 map in order to the 38 PNG ordinals 1..38. This ordinal correspondence is supported by counts (results/window_lists.json, written by scripts/check_window_lists.py), but the original temporal provenance is unverified. Gait step_frames=20 cannot generate those retained indices within 276 frames.

GEI uses nonempty centroid-aligned silhouettes; CGI uses equal-index thirds of that alignment. Alignment rounds centroids to pixels and crops the union of centered foreground bounds before resizing. The unknown MATLAB morphology and body-proportion crop are not emulated.
FDEI uses segment-peak-retained static energy, adds positive previous-minus-current differences, and sets the first previous frame equal to the first current frame. The Methods give no numerical denoising threshold. HEI averages whole/front/rear masks in one trajectory union; front direction is inferred from net centroid motion, and comb6 is rear/front/whole RGB.

What matches and what does not: HEI_comb6 r = 0.896, MAE = 0.076; FDEI r = 0.871, MAE = 0.220; GEI r = 0.863, MAE = 0.110; CGI r = 0.786, MAE = 0.151. Spatial structure agrees for every modality (all mean r at least 0.786); agreement is best for HEI_comb6 and weakest in intensity for FDEI (largest MAE), whose archived images peak at 0.52 while the regenerated ones peak at 1.00, consistent with a different weighting or normalization of the static term in the pipeline that produced the archived images, which cannot be inspected.
Identity is not expected: the archived energy images were produced by the MATLAB stage with morphological cleaning and a body-proportion crop that are not part of these equations, and the window hop is not recorded in the metadata. These measurements characterize a runnable reference implementation of the equations; the main anomaly analysis uses the archived PNGs unchanged.

Per-window values: results/ei_xcheck.csv. results/ei_xcheck.json carries the pass's mask_meta.json as a verbatim copy under `metadata` (its `note` field is in Chinese and describes the mask export stage). Generated PNGs: results/ei_regen_check/ (1,368 files, versioned with this repository).

## Second keep-list: `keep_windows_gated`

Same pass, window length and hop; 38 retained metadata indices 4..41 map in order to PNG ordinals 1..38. Per-window values: results/ei_xcheck_gated.csv; summary: results/ei_xcheck_gated.json.

| Modality | Mean window MAE [0,1] | Mean window Pearson r | Archived mean / max | Regenerated mean / max |
|---|---:|---:|---:|---:|
| CGI | 0.137192 | 0.808799 | 0.469 / 1.000 | 0.501 / 1.000 |
| FDEI | 0.220779 | 0.871295 | 0.204 / 0.521 | 0.395 / 1.000 |
| GEI | 0.107306 | 0.864827 | 0.469 / 1.000 | 0.501 / 1.000 |
| HEI_comb6 | 0.057489 | 0.947696 | 0.225 / 1.000 | 0.241 / 1.000 |

For this pass, `keep_windows_gated` gives the higher mean Pearson r in 4 of 4 compared modalities and the lower mean MAE in 3 of 4 than `keep_windows_w30` (mean r over the compared modalities 0.873 versus 0.854).

## Per-window values (`keep_windows_w30`)

| PNG ordinal | GEI MAE | GEI r | CGI MAE | CGI r | FDEI MAE | FDEI r | HEI_comb6 MAE | HEI_comb6 r |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1047 | 0.8768 | 0.1595 | 0.7778 | 0.2308 | 0.8749 | 0.0871 | 0.8674 |
| 2 | 0.0997 | 0.8792 | 0.1494 | 0.7921 | 0.2290 | 0.8839 | 0.0858 | 0.8789 |
| 3 | 0.1015 | 0.8725 | 0.1518 | 0.7857 | 0.2354 | 0.8781 | 0.0842 | 0.8829 |
| 4 | 0.1098 | 0.8635 | 0.1489 | 0.7882 | 0.2339 | 0.8800 | 0.0814 | 0.8808 |
| 5 | 0.1121 | 0.8643 | 0.1489 | 0.7878 | 0.2343 | 0.8773 | 0.0757 | 0.9074 |
| 6 | 0.1039 | 0.8748 | 0.1457 | 0.8015 | 0.2271 | 0.8751 | 0.0703 | 0.9068 |
| 7 | 0.1050 | 0.8703 | 0.1454 | 0.7968 | 0.2273 | 0.8758 | 0.0720 | 0.9061 |
| 8 | 0.1083 | 0.8704 | 0.1524 | 0.7877 | 0.2220 | 0.8795 | 0.0750 | 0.8954 |
| 9 | 0.0992 | 0.8855 | 0.1429 | 0.8080 | 0.2292 | 0.8894 | 0.0761 | 0.8969 |
| 10 | 0.0958 | 0.8927 | 0.1376 | 0.8151 | 0.2245 | 0.8967 | 0.0715 | 0.9136 |
| 11 | 0.0970 | 0.8912 | 0.1388 | 0.8161 | 0.2193 | 0.8934 | 0.0715 | 0.9090 |
| 12 | 0.1194 | 0.8503 | 0.1506 | 0.7876 | 0.2281 | 0.8672 | 0.0711 | 0.9074 |
| 13 | 0.1168 | 0.8526 | 0.1538 | 0.7844 | 0.2269 | 0.8592 | 0.0753 | 0.8925 |
| 14 | 0.1120 | 0.8604 | 0.1410 | 0.8086 | 0.2206 | 0.8733 | 0.0754 | 0.9003 |
| 15 | 0.1158 | 0.8447 | 0.1550 | 0.7790 | 0.2312 | 0.8456 | 0.0697 | 0.9123 |
| 16 | 0.1200 | 0.8410 | 0.1541 | 0.7749 | 0.2379 | 0.8572 | 0.0704 | 0.9162 |
| 17 | 0.1331 | 0.8198 | 0.1666 | 0.7546 | 0.2417 | 0.8506 | 0.0698 | 0.9142 |
| 18 | 0.1288 | 0.8292 | 0.1661 | 0.7577 | 0.2345 | 0.8544 | 0.0670 | 0.9115 |
| 19 | 0.1205 | 0.8489 | 0.1547 | 0.7760 | 0.2335 | 0.8662 | 0.0698 | 0.9096 |
| 20 | 0.1149 | 0.8553 | 0.1514 | 0.7857 | 0.2251 | 0.8695 | 0.0687 | 0.9125 |
| 21 | 0.1118 | 0.8624 | 0.1431 | 0.8015 | 0.2192 | 0.8796 | 0.0678 | 0.9212 |
| 22 | 0.1150 | 0.8654 | 0.1479 | 0.7896 | 0.2210 | 0.8836 | 0.0672 | 0.9213 |
| 23 | 0.1071 | 0.8803 | 0.1356 | 0.8157 | 0.2151 | 0.8978 | 0.0640 | 0.9194 |
| 24 | 0.0902 | 0.9024 | 0.1288 | 0.8367 | 0.2100 | 0.8998 | 0.0622 | 0.9171 |
| 25 | 0.0958 | 0.8893 | 0.1403 | 0.8099 | 0.2110 | 0.8963 | 0.0693 | 0.9106 |
| 26 | 0.0989 | 0.8799 | 0.1470 | 0.7934 | 0.2167 | 0.8851 | 0.0686 | 0.9096 |
| 27 | 0.1041 | 0.8639 | 0.1532 | 0.7782 | 0.2218 | 0.8666 | 0.0767 | 0.8872 |
| 28 | 0.1048 | 0.8586 | 0.1483 | 0.7872 | 0.2158 | 0.8695 | 0.0720 | 0.9000 |
| 29 | 0.1192 | 0.8320 | 0.1642 | 0.7527 | 0.2279 | 0.8325 | 0.0681 | 0.9133 |
| 30 | 0.1277 | 0.8175 | 0.1659 | 0.7506 | 0.2244 | 0.8180 | 0.0815 | 0.8983 |
| 31 | 0.1168 | 0.8404 | 0.1521 | 0.7761 | 0.2121 | 0.8425 | 0.0890 | 0.8915 |
| 32 | 0.1116 | 0.8588 | 0.1463 | 0.7907 | 0.2078 | 0.8597 | 0.0851 | 0.8917 |
| 33 | 0.1059 | 0.8710 | 0.1526 | 0.7782 | 0.2038 | 0.8697 | 0.0875 | 0.8802 |
| 34 | 0.0987 | 0.8815 | 0.1428 | 0.8038 | 0.1943 | 0.8882 | 0.0972 | 0.8528 |
| 35 | 0.1125 | 0.8664 | 0.1659 | 0.7518 | 0.1933 | 0.8671 | 0.1025 | 0.8370 |
| 36 | 0.1109 | 0.8671 | 0.1599 | 0.7761 | 0.1852 | 0.8666 | 0.0889 | 0.8612 |
| 37 | 0.1136 | 0.8609 | 0.1677 | 0.7574 | 0.1865 | 0.8605 | 0.0841 | 0.8666 |
| 38 | 0.1141 | 0.8595 | 0.1630 | 0.7607 | 0.1896 | 0.8603 | 0.0803 | 0.8615 |

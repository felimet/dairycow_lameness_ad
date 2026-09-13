# Cow lameness anomaly detection: analysis code

*Traditional Chinese version: [README.zh-TW.md](README.zh-TW.md)*

Analysis code implementing the methods described in the manuscript *Reliability-aware anomaly detection of dairy cow lameness from side-view gait and energy images under commercial-farm crowding*, under review at *Scientific Reports*. The manuscript is not published; nothing here should be read as a published result. The package `cowlame` implements energy-image synthesis from binary silhouettes (GEI, CGI, FDEI, HEI), four one-class anomaly detectors (PCA subspace, PatchCore, convolutional autoencoder, STFPM) fitted only on normal passes, cow-independent five-fold `GroupKFold` evaluation with pass-level AUROC/AUPRC/max-F1 and t-based 95% intervals, a label-permutation test, severity and crowding analyses, and the design-factor analyses (full image versus leg-region crop, HEI channel permutations, late fusion, per-cow score normalization). A separate script validates the upstream YOLO11m-seg model on both dataset splits.

Running the scripts produces pass-level out-of-fold scores for every configuration, per-fold and pooled metrics, training curves, figure-data CSVs, and `results/REPORT.md`, a side-by-side table of the values obtained here against the values reported in the manuscript.

## What the scripts do

| Script | Purpose | Main outputs |
|---|---|---|
| `scripts/make_cohort.py` | Cohort counts from the aligned manifest (windows, passes, known cows, `GroupKFold` groups, consensus and crowding counts, per-fold pass counts), cross-checked against the pass-level expert-score file, with SHA-256 of both CSVs | `results/cohort.json` |
| `scripts/run_main.py` | Four detectors x four energy-image modalities, leg-region crop, cow-independent folds | `results/runs/*_oof.csv`, `results/oof_scores.csv`, `results/convae_curves.csv`, `results/stfpm_curves.csv`, `results/runtime_main.json` |
| `scripts/run_ablations.py` | Full-image versus leg-crop table, six HEI channel permutations, late fusion (Subspace on concatenated pass means; PatchCore on concatenated patch features), per-cow normalization | `results/table2_legcrop.csv` (`full_auroc`, `leg_auroc`, `delta` are fold means; `full_pooled_auroc`, `leg_pooled_auroc`, `pooled_delta` are the pooled out-of-fold estimand), `results/hei_perm.csv` (`mean`, `ci_low`, `ci_high` fold-based, `pooled_auroc` pooled), `results/fusion_percow.json`, `results/percow_scores.csv` |
| `scripts/make_report.py` | Metrics, permutation tests, severity and crowding analyses, comparison with manuscript values, runtime summary | `results/REPORT.md`, `results/RUNTIME.md`, `results/summary.json`, `results/permutation.json`, `results/severity.json`, `results/crowding.csv`, `results/fig4_bars.csv` (`mean`, `ci_low`, `ci_high` are fold-based; `pooled_auroc` is the pooled out-of-fold estimand), `results/fig5_crowding.csv` |
| `scripts/regenerate_ei.py` | Energy images from per-frame silhouettes with `cowlame/energy_images.py`, cross-checked against the archived images of one pass (`--window-list w30` or `gated` selects the `mask_meta.json` keep-list; `--compare-list` runs the other list as well and appends its table) | `docs/EI_XCHECK.md`, `results/ei_xcheck.csv`, `results/ei_xcheck.json`, `results/ei_xcheck_<list>.csv/.json` for the compared list, `results/ei_regen_check/` (PNGs, not versioned) |
| `scripts/check_window_lists.py` | Manifest window count per pass against the keep-lists and stride period recorded in `mask_meta.json` | `results/window_lists.json` |
| `scripts/segmentation_xcheck.py` | Ultralytics validation of the trained YOLO11m-seg weights on the `val` and `test` splits, compared with manuscript Table 1 and with the retained-checkpoint values reported in the Results (`--render-only` rewrites the comparison from the stored records) | `docs/SEG_XCHECK.md`, `results/segmentation_xcheck.json` |
| `scripts/freeze_environment.py` | Pins the installed package versions (prints a diff against the tracked files by default; `--write` replaces them; conda build-host `file://` entries are pinned by their installed version; the raw `pip freeze` output is kept in `results/pip_freeze_full.txt`) | `requirements.txt`, `environment.yml`, `results/pip_freeze_full.txt` |
| `scripts/verify_results.py` | Acceptance check of the deposited `results/` (file presence, grid completeness, cohort counts against `results/cohort.json`, cow-disjoint folds, curve lengths, permutation bookkeeping, segmentation hashes); no retraining | pass/fail |

Package modules: `cowlame/config.py` (all fixed settings, logged at the start of a run by every script except `scripts/check_window_lists.py`; `scripts/run_main.py`, `scripts/run_ablations.py` and `scripts/segmentation_xcheck.py` log them from `setup_runtime`, the other scripts from their own entry point), `cowlame/data.py` (manifest ingestion, path remapping, crop/resize, tensor cache), `cowlame/detectors/` (`subspace`, `patchcore`, `convae`, `stfpm`; each exposes `fit`, `score`, `localize`), `cowlame/evaluate.py` (folds, metrics, intervals, out-of-fold bookkeeping), `cowlame/stats.py` (permutation test, severity, crowding, per-cow normalization), `cowlame/energy_images.py` (Eq. 1 to 5), `cowlame/manuscript_reference.json` (manuscript values used for the comparison table, each block labelled by its source: exact text, figure reading, or not reported).

## Data layout

Energy images, silhouettes, videos, labels and segmentation weights are not distributed with the code; they are available from the corresponding author. The analysis reads a data root with this layout (default `F:/cow-data/01_data`, overridable with `--data-root` or `COWLAME_DATA_ROOT`):

```text
<data-root>/
  window_manifest_maskfirst.csv          one row per gait-cycle window; paths to the energy images (read by every analysis script)
  gt_passes_001_118.csv                  expert locomotion scores per pass (read by make_cohort.py only, for the consensus cross-check)
  regen_alignment.csv                    per-recording alignment status (present in the data root; not read by any script)
  matlab_ei/<rec:03d>/<pass:02d>/<MOD>/<MOD>_<rec:03d>_<pass:02d>_<win:03d>.png   MOD in GEI, CGI, FDEI, HEI_comb1..6
  masks/<rec:03d>/<pass:02d>/mask_meta.json and frames/frame_*.png         per-frame silhouettes (used by regenerate_ei.py)
```

`cowlame/data.py` requires the manifest columns `aligned` (rows whose value is `True`, case-insensitive, are kept), `recording`, `pass`, `window`, `cow_id` (empty for unknown identity), `score_1_3`, `max_concurrent`, and one path column per modality (`gei_path`, `cgi_path`, `fdei_path`, `hei_comb1_path` to `hei_comb6_path`; modality HEI is read from `hei_comb6_path`). The manifest's `anomaly` and `hei_path` columns are not read; `video_name` is read only by `scripts/make_cohort.py` for the consensus cross-check. `scripts/regenerate_ei.py` reads `window_size`, `keep_windows_w30` or `keep_windows_gated`, and `gait.step_frames` from `mask_meta.json`. `cowlame.data.remap_path` discards everything up to and including the path component `01_data` of a manifest path and joins the remainder onto the data root; paths without that component are taken relative to the data root; every resulting file must exist. The analysis uses the aligned rows of the manifest; the binary label is 1 for consensus score 2 or 3 and 0 for score 1; crowding is `max_concurrent` clipped at 3. Passes without a cow ID form one pseudo-group `unknown` for `GroupKFold` (`--unknown-cow-policy per-pass` treats each as its own group, for a sensitivity check written to a separate `--out`). `scripts/make_cohort.py` writes the cohort counts obtained from the manifest to `results/cohort.json`, which `scripts/make_report.py` summarizes in `results/REPORT.md`.

The upstream segmentation stage is documented in `upstream/segmentation/` (training arguments and notebooks of the YOLO11m-seg model).

## Installation

Python 3.11 with CUDA-enabled PyTorch is required for the analyses (the scripts refuse to fall back to CPU). The pinned versions actually used are in `requirements.txt` (pip) and `environment.yml` (conda, with the PyTorch CUDA 12.8 wheel index).

```powershell
conda env create -f environment.yml          # or: conda create -y --override-channels -c conda-forge -n cowlame python=3.11.16 && conda run -n cowlame python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
conda run -n cowlame python -m pip install -e . --no-deps --no-build-isolation
conda run -n cowlame python -c "import torch; print(torch.cuda.is_available())"
```

If conda reports unaccepted Terms of Service for the Anaconda `defaults` channels, run `conda tos accept` or use `--override-channels` as above. Only `scripts/run_main.py`, `scripts/run_ablations.py` and `scripts/segmentation_xcheck.py` call `setup_runtime` and therefore create the gitignored `cache/` folder inside the repository (tensor cache, backbone weights, temporary files). Writing outputs is separate: every script except `scripts/verify_results.py` writes files under `--out`; `scripts/regenerate_ei.py` and `scripts/segmentation_xcheck.py` additionally write `docs/EI_XCHECK.md` and `docs/SEG_XCHECK.md`; `scripts/freeze_environment.py --write` replaces the tracked `requirements.txt` and `environment.yml` and writes `results/pip_freeze_full.txt`. `scripts/verify_results.py` reads only. On Windows, `conda run` needs an ASCII temporary directory when the repository path contains non-ASCII characters; `reproduce.ps1` sets `TEMP`/`TMP` to a `tmp` folder inside the environment. Alternatively call the environment interpreter directly (`<env>/python.exe scripts/...`).

## Reproduction

```powershell
./reproduce.ps1 -DataRoot F:/cow-data/01_data -Out results      # Windows; chains the first four commands below
bash reproduce.sh results                                        # POSIX; reads COWLAME_DATA_ROOT
```

```powershell
conda run -n cowlame python scripts/make_cohort.py --data-root F:/cow-data/01_data --out results
conda run -n cowlame python scripts/run_main.py --data-root F:/cow-data/01_data --out results
conda run -n cowlame python scripts/run_ablations.py --data-root F:/cow-data/01_data --out results
conda run -n cowlame python scripts/make_report.py --out results
conda run -n cowlame python scripts/regenerate_ei.py --data-root F:/cow-data/01_data --out results --recording 1 --pass-id 1 --window-list w30 --compare-list gated
conda run -n cowlame python scripts/check_window_lists.py --data-root F:/cow-data/01_data --out results
conda run -n cowlame python scripts/segmentation_xcheck.py --weights <path>/yolo11m-seg_cow/weights/best.pt --dataset <path>/dataset_seg/yolo_data_integrated/dataset.yaml --out results
conda run -n cowlame python scripts/verify_results.py --out results
conda run -n cowlame python scripts/freeze_environment.py --write
```

Completed configurations are reused when their configuration/manifest signature matches; use `--force` after changing settings, or a new `--out`. Image tensors are cached under `cache/` keyed by manifest hash; the pretrained MobileNetV3 backbone is downloaded into `cache/torch`. The segmentation script writes a repository-local copy of the dataset YAML with an absolute data root, keeps label caches under `cache/segmentation`, and installs an audit hook that rejects file writes outside the repository and the conda environment.

## Re-execution versus manuscript values

Values reported in the manuscript were obtained on a GTX 1080 8 GB / CUDA 11.8 / PyTorch 2.5.1 system. Re-execution with this repository on an RTX 5070 / CUDA 12.8 / PyTorch 2.11 system yields the values in [results/REPORT.md](results/REPORT.md), which tabulates every comparable quantity next to its manuscript value with the absolute difference and lists the AUROC differences larger than 0.03. The differences are consistent with library-version, GPU-numerics and random-sampling effects; no setting was tuned toward a manuscript value, and no manuscript value is claimed to be an output of this exact software environment. Quantities the manuscript reports only approximately or only in a figure are labelled as such in the table. Anything not executed is marked "not run" rather than filled in. One difference concerns a reported p value rather than a metric: the label-permutation test reproduces the PatchCore CGI leg-crop result (0 exceedances in 10,000 permutations; manuscript p < 0.001) but returns 37 exceedances in 10,000 for the Subspace CGI leg-crop configuration, that is p = 0.0037 (0.0038 with the (k+1)/(B+1) correction), whereas the manuscript reports p < 0.001 from the original analysis run. `results/REPORT.md` lists this under "Permutation p values not below the manuscript's <0.001".

All seeds are fixed (42), cuDNN runs in deterministic mode with benchmarking disabled and matmul TF32 disabled (cuDNN convolution TF32 is left at the PyTorch default, enabled), and PatchCore bank sampling uses a seeded generator. Residual nondeterminism of GPU reductions and AMP can still shift AUROC at the second decimal between machines.

## Hardware and runtime

Measured wall times of every configuration and stage are generated from the JSON files in `results/` into [results/RUNTIME.md](results/RUNTIME.md) by `scripts/make_report.py`; the GPU, PyTorch and CUDA versions used are recorded in `results/runtime_main.json`. The first run also reads and caches the energy-image PNGs and downloads the backbone weights; re-runs reuse the caches and completed configurations.

## Interpretation notes

Choices made where the Methods description admits more than one reading, and the evidence boundaries of each analysis (pooled out-of-fold scores, oracle max-F1 threshold, label-permutation scope, transductive per-cow normalization, unadjusted crowding strata), are listed in [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md). The energy-image cross-check is in [docs/EI_XCHECK.md](docs/EI_XCHECK.md) and the segmentation cross-check in [docs/SEG_XCHECK.md](docs/SEG_XCHECK.md).

## Citation and license

Cite the software with `CITATION.cff` together with the manuscript. Release v0.1.0 is deposited at Zenodo under DOI 10.5281/zenodo.22734484 (https://doi.org/10.5281/zenodo.22734484), and this repository is its mirror. `.zenodo.json` carries the deposit metadata (creators: Zhou JM, Chu WL, Chiang HI, Paudyal S). The code is MIT licensed, copyright 2026 Jia-Ming Zhou, except the archived upstream material in `upstream/segmentation/`, which is distributed under the Apache License 2.0 (`upstream/segmentation/LICENSE`). Third-party frameworks retain their own licenses. Code written by Jia-Ming Zhou (`pyproject.toml`, `LICENSE`); the manuscript authors are listed as creators in `CITATION.cff` and `.zenodo.json`.

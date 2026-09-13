# 乳牛跛行單類別異常檢測：分析程式碼

*英文原版：README.md（兩版內容不一致時以英文版為準）*

本分析程式碼實作稿件 *Reliability-aware anomaly detection of dairy cow lameness from side-view gait and energy images under commercial-farm crowding* 所述之方法，該稿件正於 *Scientific Reports* 審查中。稿件尚未發表，此處內容均不應視為已發表結果。套件 `cowlame` 實作自二值輪廓合成能量影像（GEI、CGI、FDEI、HEI）、僅以正常通過段（pass）擬合的四種單類別（one-class）異常檢測器（PCA subspace、PatchCore、卷積自編碼器、STFPM）、依牛隻分組的五摺 `GroupKFold` 評估（含通過段層級的 AUROC/AUPRC/max-F1 與基於 t 分布的 95% 區間）、標籤置換檢定、嚴重度與擁擠度分析，以及設計因子分析（全幅影像相對於腿部裁切、HEI 通道排列、晚期融合、逐牛分數正規化）。另有獨立腳本在兩個資料集分割（split）上驗證上游的 YOLO11m-seg 模型。

執行這些腳本會產生每一組態的通過段層級出摺（out-of-fold）分數、各摺與合併（pooled）指標、訓練曲線、圖表資料 CSV，以及 `results/REPORT.md`，後者為本處所得數值與稿件所報數值的並列對照表。

## 各腳本的功能

| 腳本 | 用途 | 主要輸出 |
|---|---|---|
| `scripts/make_cohort.py` | 自對齊後的 manifest 計算 cohort（資料群體）計數（視窗、通過段、已知牛隻、`GroupKFold` 分組、共識與擁擠度計數、各摺通過段數），並與通過段層級的專家評分檔交叉核對，附兩份 CSV 的 SHA-256 | `results/cohort.json` |
| `scripts/run_main.py` | 四種檢測器 x 四種能量影像模態、腿部裁切、依牛隻分組的摺 | `results/runs/*_oof.csv`、`results/oof_scores.csv`、`results/convae_curves.csv`、`results/stfpm_curves.csv`、`results/runtime_main.json` |
| `scripts/run_ablations.py` | 全幅影像相對於腿部裁切的表格、六種 HEI 通道排列、晚期融合（Subspace 作用於串接的通過段平均；PatchCore 作用於串接的 patch 特徵）、逐牛正規化 | `results/table2_legcrop.csv`（`full_auroc`、`leg_auroc`、`delta` 為各摺平均；`full_pooled_auroc`、`leg_pooled_auroc`、`pooled_delta` 為合併出摺被估計量）、`results/hei_perm.csv`（`mean`、`ci_low`、`ci_high` 基於摺，`pooled_auroc` 為合併值）、`results/fusion_percow.json`、`results/percow_scores.csv` |
| `scripts/make_report.py` | 指標、置換檢定、嚴重度與擁擠度分析、與稿件數值的比較、執行時間摘要 | `results/REPORT.md`、`results/RUNTIME.md`、`results/summary.json`、`results/permutation.json`、`results/severity.json`、`results/crowding.csv`、`results/fig4_bars.csv`（`mean`、`ci_low`、`ci_high` 基於摺；`pooled_auroc` 為合併出摺被估計量）、`results/fig5_crowding.csv` |
| `scripts/regenerate_ei.py` | 以 `cowlame/energy_images.py` 自逐畫格輪廓產生能量影像，並與某一通過段的封存影像交叉核對（`--window-list w30` 或 `gated` 選擇 `mask_meta.json` 中的 keep-list；`--compare-list` 額外執行另一份清單並附加其表格） | `docs/EI_XCHECK.md`、`results/ei_xcheck.csv`、`results/ei_xcheck.json`、比較清單對應的 `results/ei_xcheck_<list>.csv/.json`、`results/ei_regen_check/`（PNG，不納入版控） |
| `scripts/check_window_lists.py` | 將 manifest 中各通過段的視窗數與 `mask_meta.json` 所記錄的 keep-list 及步幅週期（stride period）核對 | `results/window_lists.json` |
| `scripts/segmentation_xcheck.py` | 以 Ultralytics 在 `val`（驗證集）與 `test`（測試集）分割上驗證已訓練的 YOLO11m-seg 權重，並與稿件 Table 1 及 Results 所報告的保留檢查點數值比較（`--render-only` 由已存紀錄重新產生比較） | `docs/SEG_XCHECK.md`、`results/segmentation_xcheck.json` |
| `scripts/freeze_environment.py` | 釘選已安裝的套件版本（預設列印與受版控追蹤檔案的差異；`--write` 會取代這些檔案；conda 建置主機的 `file://` 項目以其已安裝版本釘選；原始 `pip freeze` 輸出保存於 `results/pip_freeze_full.txt`） | `requirements.txt`、`environment.yml`、`results/pip_freeze_full.txt` |
| `scripts/verify_results.py` | 對存放的 `results/` 進行驗收檢查（檔案存在性、網格完整性、cohort 計數與 `results/cohort.json` 核對、各摺牛隻互斥、曲線長度、置換檢定簿記、分割雜湊）；不重新訓練 | pass/fail |

套件模組：`cowlame/config.py`（所有固定設定；除 `scripts/check_window_lists.py` 外，每個腳本都在執行開始時記錄這些設定；`scripts/run_main.py`、`scripts/run_ablations.py` 與 `scripts/segmentation_xcheck.py` 自 `setup_runtime` 記錄，其餘腳本自各自的進入點記錄）、`cowlame/data.py`（manifest 讀入、路徑重新對應、裁切/縮放、張量快取）、`cowlame/detectors/`（`subspace`、`patchcore`、`convae`、`stfpm`；每個都提供 `fit`、`score`、`localize`）、`cowlame/evaluate.py`（摺、指標、區間、出摺簿記）、`cowlame/stats.py`（置換檢定、嚴重度、擁擠度、逐牛正規化）、`cowlame/energy_images.py`（Eq. 1 至 5）、`cowlame/manuscript_reference.json`（比較表所用的稿件數值，每個區塊皆標註其來源：原文精確數值、圖上讀值或未報告）。

## 資料配置

能量影像、輪廓、影片、標籤與分割權重不隨程式碼散布；可向通訊作者索取。分析程式讀取具下列配置的資料根目錄（預設為 `F:/cow-data/01_data`，可以 `--data-root` 或 `COWLAME_DATA_ROOT` 覆寫）：

```text
<data-root>/
  window_manifest_maskfirst.csv          one row per gait-cycle window; paths to the energy images (read by every analysis script)
  gt_passes_001_118.csv                  expert locomotion scores per pass (read by make_cohort.py only, for the consensus cross-check)
  regen_alignment.csv                    per-recording alignment status (present in the data root; not read by any script)
  matlab_ei/<rec:03d>/<pass:02d>/<MOD>/<MOD>_<rec:03d>_<pass:02d>_<win:03d>.png   MOD in GEI, CGI, FDEI, HEI_comb1..6
  masks/<rec:03d>/<pass:02d>/mask_meta.json and frames/frame_*.png         per-frame silhouettes (used by regenerate_ei.py)
```

`cowlame/data.py` 需要 manifest 具備下列欄位：`aligned`（值為 `True` 的列會被保留，不區分大小寫）、`recording`、`pass`、`window`、`cow_id`（身分未知時為空）、`score_1_3`、`max_concurrent`，以及每種模態各一個路徑欄位（`gei_path`、`cgi_path`、`fdei_path`、`hei_comb1_path` 至 `hei_comb6_path`；HEI 模態自 `hei_comb6_path` 讀取）。manifest 的 `anomaly` 與 `hei_path` 欄位不會被讀取；`video_name` 僅由 `scripts/make_cohort.py` 讀取以進行共識交叉核對。`scripts/regenerate_ei.py` 自 `mask_meta.json` 讀取 `window_size`、`keep_windows_w30` 或 `keep_windows_gated`，以及 `gait.step_frames`。`cowlame.data.remap_path` 會捨棄 manifest 路徑中直到並包含路徑元件 `01_data` 的所有部分，並將其餘部分接到資料根目錄之後；不含該元件的路徑則視為相對於資料根目錄；每個由此得到的檔案都必須存在。分析使用 manifest 中已對齊的列；二元標籤在共識分數為 2 或 3 時為 1，分數為 1 時為 0；擁擠度為 `max_concurrent` 於 3 截斷後的值。沒有牛隻 ID 的通過段在 `GroupKFold` 中構成單一虛擬群組 `unknown`（`--unknown-cow-policy per-pass` 將每個通過段視為獨立群組，用於寫入另一個 `--out` 的敏感度檢查）。`scripts/make_cohort.py` 將自 manifest 取得的 cohort 計數寫入 `results/cohort.json`，`scripts/make_report.py` 再將其摘要於 `results/REPORT.md`。

上游分割階段記載於 `upstream/segmentation/`（YOLO11m-seg 模型的訓練引數與 notebook）。

## 安裝

分析需要 Python 3.11 與啟用 CUDA 的 PyTorch（腳本拒絕退回 CPU 執行）。實際使用的釘選版本記載於 `requirements.txt`（pip）與 `environment.yml`（conda，含 PyTorch CUDA 12.8 wheel 索引）。

```powershell
conda env create -f environment.yml          # or: conda create -y --override-channels -c conda-forge -n cowlame python=3.11.16 && conda run -n cowlame python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
conda run -n cowlame python -m pip install -e . --no-deps --no-build-isolation
conda run -n cowlame python -c "import torch; print(torch.cuda.is_available())"
```

若 conda 回報 Anaconda `defaults` 頻道的服務條款尚未接受，執行 `conda tos accept` 或如上使用 `--override-channels`。只有 `scripts/run_main.py`、`scripts/run_ablations.py` 與 `scripts/segmentation_xcheck.py` 會呼叫 `setup_runtime`，因此會在儲存庫內建立已列入 gitignore 的 `cache/` 資料夾（張量快取、backbone 權重、暫存檔）。輸出的寫入則是另一回事：除 `scripts/verify_results.py` 外，每個腳本都在 `--out` 下寫入檔案；`scripts/regenerate_ei.py` 與 `scripts/segmentation_xcheck.py` 另外寫入 `docs/EI_XCHECK.md` 與 `docs/SEG_XCHECK.md`；`scripts/freeze_environment.py --write` 會取代受版控追蹤的 `requirements.txt` 與 `environment.yml`，並寫入 `results/pip_freeze_full.txt`。`scripts/verify_results.py` 只讀取。在 Windows 上，當儲存庫路徑含非 ASCII 字元時，`conda run` 需要 ASCII 的暫存目錄；`reproduce.ps1` 會將 `TEMP`/`TMP` 設為環境內的 `tmp` 資料夾。或者直接呼叫該環境的直譯器（`<env>/python.exe scripts/...`）。

## 重現

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

已完成的組態在其組態/manifest 簽章相符時會被重用；變更設定後請使用 `--force`，或改用新的 `--out`。影像張量以 manifest 雜湊為鍵快取於 `cache/` 之下；預訓練的 MobileNetV3 backbone 會下載至 `cache/torch`。分割腳本會在儲存庫內寫入一份含絕對資料根目錄的資料集 YAML 複本，將標籤快取保存於 `cache/segmentation` 之下，並安裝一個稽核 hook，拒絕對儲存庫與 conda 環境以外位置的檔案寫入。

## 重新執行結果與稿件數值的比較

稿件所報數值係於 GTX 1080 8 GB / CUDA 11.8 / PyTorch 2.5.1 系統上取得。以本儲存庫在 RTX 5070 / CUDA 12.8 / PyTorch 2.11 系統上重新執行，所得數值列於 [results/REPORT.md](results/REPORT.md)，該檔將每個可比較的量與其稿件數值並列並附絕對差異，並列出大於 0.03 的 AUROC 差異。這些差異與函式庫版本、GPU 數值運算與隨機取樣效應相符；沒有任何設定朝稿件數值調校，也不宣稱任何稿件數值是此一確切軟體環境的輸出。稿件僅以近似值或僅於圖中報告的量，在表中均標註為此類。未執行的項目標記為「not run」而非填入數值。有一項差異涉及所報的 p 值而非指標：標籤置換檢定重現了 PatchCore CGI 腿部裁切的結果（10,000 次置換中 0 次超過；稿件 p < 0.001），但對 Subspace CGI 腿部裁切組態回傳 10,000 次中 37 次超過，即 p = 0.0037（採 (k+1)/(B+1) 校正後為 0.0038），而稿件依原始分析執行報告 p < 0.001。`results/REPORT.md` 將此列於「Permutation p values not below the manuscript's <0.001」之下。

所有種子皆固定（42），cuDNN 以確定性模式執行，停用 benchmarking 並停用 matmul TF32（cuDNN 卷積 TF32 維持 PyTorch 預設，即啟用），PatchCore 記憶庫取樣使用帶種子的產生器。GPU 歸約與 AMP 的殘餘非確定性仍可能使 AUROC 在不同機器間於小數點第二位變動。

## 硬體與執行時間

每一組態與階段的實測牆鐘時間由 `scripts/make_report.py` 自 `results/` 中的 JSON 檔產生至 [results/RUNTIME.md](results/RUNTIME.md)；所用的 GPU、PyTorch 與 CUDA 版本記錄於 `results/runtime_main.json`。首次執行還會讀取並快取能量影像 PNG，並下載 backbone 權重；再次執行則重用快取與已完成的組態。

## 詮釋說明

當 Methods 描述容許多於一種解讀時所做的選擇，以及各分析的證據邊界（合併的出摺分數、oracle max-F1 閾值、標籤置換的範圍、轉導式（transductive）逐牛正規化、未校正的擁擠度分層），列於 [docs/IMPLEMENTATION_NOTES.zh-TW.md](docs/IMPLEMENTATION_NOTES.zh-TW.md)。能量影像交叉核對見 [docs/EI_XCHECK.zh-TW.md](docs/EI_XCHECK.zh-TW.md)，分割交叉核對見 [docs/SEG_XCHECK.zh-TW.md](docs/SEG_XCHECK.zh-TW.md)。

## 引用與授權

引用本軟體時請以 `CITATION.cff` 連同稿件一併引用。`.zenodo.json` 載有存放的 metadata（creators：Zhou JM、Chu WL、Chiang HI、Paudyal S）。程式碼採 MIT 授權，版權所有 2026 Jia-Ming Zhou，但 `upstream/segmentation/` 中封存的上游素材除外，其依 Apache License 2.0 散布（`upstream/segmentation/LICENSE`）。第三方框架保留其各自的授權。程式碼由 Jia-Ming Zhou 撰寫（`pyproject.toml`、`LICENSE`）；稿件作者列為 `CITATION.cff` 與 `.zenodo.json` 中的 creators。

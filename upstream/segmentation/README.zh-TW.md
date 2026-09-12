# 上游階段 A：YOLO11m-seg 實例分割（參考材料）

*英文原版：upstream/segmentation/README.md（兩版內容不一致時以英文版為準）*

本資料夾為能量影像合成之前的分割 / 追蹤階段的參考材料，複製自同一作者群封存的資料集 / 模型套件。Notebook 的輸出已以純 JSON 改寫的方式清除（`outputs` 清空、執行計數重設）。

| 檔案 | 來源 | 內容 |
|---|---|---|
| `args.yaml` | `use_model/yolo11m-seg_cow/args.yaml` | 封存權重之 Ultralytics 訓練參數的權威紀錄（`yolo11m-seg`、imgsz 512、batch 32、100 epochs、patience 25、AdamW、seed 42、`split: val`）。 |
| `train_yolo11_seg.ipynb` | `train/train_yolo11_seg.ipynb` | 訓練 notebook（Ultralytics Python API）。其 `train()` 呼叫在 `warmup_momentum`（0.0001 對 1.0e-05）、`warmup_bias_lr`（0.005 對 0.01）及專案名稱（`seg_training_result` 對 `training_result_yolo11m-seg`）上與 `args.yaml` 不一致；兩者有差異時以 `args.yaml` 為準。Comet ML 登入儲存格內為佔位憑證，執行前須先編輯或移除。 |
| `inference_seg.ipynb` | `use_model/inference_seg.ipynb` | 以訓練完成的權重執行單張影像推論的範例。 |

權重（`best.pt`）、已標註資料集（驗證集與測試集的影像數量見 `docs/SEG_XCHECK.md`；訓練集未在此計數）與影片均未收錄，可向通訊作者索取。本資料夾內的檔案依 Apache License 2.0 散布（見本資料夾的 `LICENSE`），有別於儲存庫其餘部分所採用的 MIT 授權。

等效的 Ultralytics 命令列（依 `args.yaml`）：

```powershell
# training (as archived; two GPUs, disk cache)
yolo segment train model=yolo11m-seg.pt data=dataset_seg/yolo_data_integrated/dataset.yaml imgsz=512 batch=32 epochs=100 patience=25 optimizer=AdamW lr0=0.005 lrf=0.0001 weight_decay=0.0007 warmup_epochs=3 cos_lr=True dropout=0.15 mosaic=0.8 mixup=0.5 cutmix=0.5 copy_paste=0.3 translate=0.15 scale=0.3 seed=42 deterministic=True device=0,1 workers=8 cache=disk project=training_result_yolo11m-seg name=yolo11m-seg_cow
# validation on a named split (what scripts/segmentation_xcheck.py does programmatically for val and test)
yolo segment val model=best.pt data=dataset.yaml imgsz=512 split=test batch=16 conf=0.001 iou=0.7 half=False plots=False
```

驗證集與測試集兩者與稿件 Table 1 及 Results 數值的比對見 `docs/SEG_XCHECK.md`。多目標追蹤（稿件採用 ByteTrack；`tracker: botsort.yaml` 僅為 `args.yaml` 所記錄的 Ultralytics 預設值）未在這些 notebook 中執行，故無法自本材料驗證。

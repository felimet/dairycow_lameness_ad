# 分割交叉核對

*英文原版：docs/SEG_XCHECK.md（兩版內容不一致時以英文版為準；英文版由 `scripts/segmentation_xcheck.py --render-only` 自已存紀錄產生）*

稿件對 YOLOv11m-seg 模型報告三組指標。Table 1 為訓練框架在最終訓練 epoch 記錄的驗證集指標（n = 1,101 張影像）。Results 另補充保留之最佳 fitness 檢查點（第 100 個 epoch 中的第 96 個；用於輪廓擷取並隨程式碼存放）的驗證集紀錄值，以及該檢查點在 552 張影像之保留測試集上的評估結果，後者即由本腳本產生。本腳本以保留檢查點重新驗證兩個資料分割。

| 來源 | 資料分割 | 影像數 | 類型 | P | R | mAP50 | mAP50-95 |
|---|---|---:|---|---:|---:|---:|---:|
| 稿件 Table 1（驗證集，最終 epoch 100） | val | 1101 | mask | 0.992 | 0.994 | 0.995 | 0.839 |
| 稿件 Table 1（驗證集，最終 epoch 100） | val | 1101 | box | NA | NA | 0.995 | 0.990 |
| 稿件 Results，保留檢查點（epoch 96），驗證集紀錄值 | val | 1101 | mask | 0.993 | 0.996 | 0.995 | 0.841 |
| 稿件 Results，保留檢查點（epoch 96），測試集 | test | 552 | mask | 0.989 | 0.992 | 0.995 | 0.850 |
| 稿件 Results，保留檢查點（epoch 96），測試集 | test | 552 | box | NA | NA | 0.995 | 0.989 |
| 本腳本，保留檢查點 | val | 1101 | mask | 0.993382 | 0.996233 | 0.994683 | 0.842821 |
| 本腳本，保留檢查點 | val | 1101 | box | 0.993382 | 0.996233 | 0.994683 | 0.988367 |
| 本腳本，保留檢查點 | test | 552 | mask | 0.989094 | 0.992391 | 0.994561 | 0.849982 |
| 本腳本，保留檢查點 | test | 552 | box | 0.989094 | 0.992391 | 0.994561 | 0.989395 |

以報告精度（小數點後 3 位）計的一致性：

- 稿件 Table 1（驗證集，最終 epoch 100）：6 項報告值中重現 2 項，平均絕對偏差 0.0016；差異為 mask precision 0.993 對 0.992、mask recall 0.996 對 0.994、mask map 0.843 對 0.839、box map 0.988 對 0.990。
- 稿件 Results，保留檢查點（epoch 96），驗證集紀錄值：4 項報告值中重現 3 項，平均絕對偏差 0.0007；差異為 mask map 0.843 對 0.841。
- 稿件 Results，保留檢查點（epoch 96），測試集：6 項報告值全部重現，平均絕對偏差 0.0003。

僅測試集該列預期完全重現，因為 Results 所報告的正是本腳本產生的評估。驗證集該列與訓練框架的兩組紀錄值比較：同一檢查點的 epoch 96 紀錄，兩者僅因驗證器版本與批次而異；以及 Table 1 背後的 epoch 100 紀錄，兩者另因 epoch 不同而異。封存的 args.yaml 記錄訓練期驗證使用 split=val，且未記錄 Ultralytics 版本。

Ultralytics 8.4.142；imgsz=512；batch=16；conf=0.001；iou=0.7；device=0；FP32；workers=0；無資料增強或繪圖。

```powershell
conda run -n cowlame python scripts/segmentation_xcheck.py --weights <path>/yolo11m-seg_cow/weights/best.pt --dataset <path>/dataset_seg/yolo_data_integrated/dataset.yaml --out results
conda run -n cowlame python scripts/segmentation_xcheck.py --render-only --out results   # 自已存紀錄重新產生英文版文件
```

原始資料集 YAML 含有一個已失效的 Linux 絕對路徑。一份暫時性的儲存庫本地 YAML 僅將資料根目錄替換為現行的絕對上層目錄。來源 YAML 與權重的 SHA-256 雜湊值在驗證前後相符。
Ultralytics 的標籤快取讀寫函式被重新導向至 cache/segmentation；cache=False 停用影像快取。一個 Python audit hook 會拒絕本儲存庫與 cowlame 環境以外的檔案變更。來源 YAML、標籤、影像與權重皆未被編輯。

完整精度數值與計時：results/segmentation_xcheck.json；稿件目標值存於 cowlame/manuscript_reference.json（`segmentation` 區塊）。

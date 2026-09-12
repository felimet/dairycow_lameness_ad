# 分割交叉核對

*英文原版：docs/SEG_XCHECK.md（兩版內容不一致時以英文版為準）*

稿件 Table 1 記載「held-out test set, n = 552」。

| 資料分割 | 影像數 | 類型 | P | R | mAP50 | mAP50-95 |
|---|---:|---|---:|---:|---:|---:|
| 稿件測試集 | 552 | mask | 0.992 | 0.994 | 0.995 | 0.839 |
| 稿件測試集 | 552 | box | NA | NA | 0.995 | 0.990 |
| val | 1101 | mask | 0.993382 | 0.996233 | 0.994683 | 0.842821 |
| val | 1101 | box | 0.993382 | 0.996233 | 0.994683 | 0.988367 |
| test | 552 | mask | 0.989094 | 0.992391 | 0.994561 | 0.849982 |
| test | 552 | box | 0.989094 | 0.992391 | 0.994561 | 0.989395 |

兩個資料分割皆無法在小數點後三位重現 Table 1 所報告的全部數值。以六項報告指標的平均絕對偏差計，val 較為接近；test 則具有報告的影像數（n = 552）。

Ultralytics 8.4.142；imgsz=512；batch=16；conf=0.001；iou=0.7；device=0；FP32；workers=0；無資料增強或繪圖。

```powershell
conda run -n cowlame python scripts/segmentation_xcheck.py --weights <path>/yolo11m-seg_cow/weights/best.pt --dataset <path>/dataset_seg/yolo_data_integrated/dataset.yaml --out results
```

原始資料集 YAML 含有一個已失效的 Linux 絕對路徑。一份暫時性的儲存庫本地 YAML 僅將資料根目錄替換為現行的絕對上層目錄。來源 YAML 與權重的 SHA-256 雜湊值在驗證前後相符。
Ultralytics 的標籤快取讀寫函式被重新導向至 cache/segmentation；cache=False 停用影像快取。一個 Python audit hook 會拒絕本儲存庫與 cowlame 環境以外的檔案變更。來源 YAML、標籤、影像與權重皆未被編輯。

完整精度數值與計時：results/segmentation_xcheck.json。封存的 args.yaml 記錄訓練期驗證使用 split=val，且未記錄 Ultralytics 版本；上述兩項評估與 Table 1 之間在小數點後第三位的差異，其量級符合驗證器版本與批次差異所預期的範圍，而上述的資料分割比較即為現有的實證證據。

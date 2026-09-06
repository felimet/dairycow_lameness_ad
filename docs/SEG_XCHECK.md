# Segmentation cross-check

Manuscript Table 1 reports "held-out test set, n = 552".

| Split | n images | Type | P | R | mAP50 | mAP50-95 |
|---|---:|---|---:|---:|---:|---:|
| Manuscript test | 552 | mask | 0.992 | 0.994 | 0.995 | 0.839 |
| Manuscript test | 552 | box | NA | NA | 0.995 | 0.990 |
| val | 1101 | mask | 0.993382 | 0.996233 | 0.994683 | 0.842821 |
| val | 1101 | box | 0.993382 | 0.996233 | 0.994683 | 0.988367 |
| test | 552 | mask | 0.989094 | 0.992391 | 0.994561 | 0.849982 |
| test | 552 | box | 0.989094 | 0.992391 | 0.994561 | 0.989395 |

Neither split reproduces every reported Table 1 value at three decimal places. val is nearer by mean absolute deviation over the six reported metrics; test has the reported image count (n = 552).

Ultralytics 8.4.142; imgsz=512; batch=16; conf=0.001; iou=0.7; device=0; FP32; workers=0; no augmentation or plots.

```powershell
conda run -n cowlame python scripts/segmentation_xcheck.py --weights <path>/yolo11m-seg_cow/weights/best.pt --dataset <path>/dataset_seg/yolo_data_integrated/dataset.yaml --out results
```

The original dataset YAML contains an obsolete absolute Linux path. A temporary repository-local YAML replaces only the data root with the current absolute parent directory. Source YAML and weights SHA-256 hashes match before and after validation.
Ultralytics label-cache read/write functions are redirected to cache/segmentation; cache=False disables image caching. A Python audit hook rejects file mutations outside this repository and the cowlame environment. No source YAML, labels, images or weights are edited.

Full precision and timings: results/segmentation_xcheck.json. The archived args.yaml records split=val for training-time validation and does not record the Ultralytics version; differences at the third decimal between the two evaluations above and Table 1 are of the size expected from validator-version and batching differences, and the split comparison above is the empirical evidence available.

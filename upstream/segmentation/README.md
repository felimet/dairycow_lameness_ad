# Upstream stage A: YOLO11m-seg instance segmentation (reference material)

*Traditional Chinese version: [README.zh-TW.md](README.zh-TW.md)*

Reference material for the segmentation/tracking stage that precedes energy-image synthesis, copied from the archived dataset/model package of the same authors. Notebook outputs were cleared (`outputs` emptied, execution counts reset) by a plain JSON rewrite.

| File | Origin | Content |
|---|---|---|
| `args.yaml` | `use_model/yolo11m-seg_cow/args.yaml` | Authoritative record of the Ultralytics training arguments of the archived weights (`yolo11m-seg`, imgsz 512, batch 32, 100 epochs, patience 25, AdamW, seed 42, `split: val`). |
| `train_yolo11_seg.ipynb` | `train/train_yolo11_seg.ipynb` | Training notebook (Ultralytics Python API). Its `train()` call deviates from `args.yaml` in `warmup_momentum` (0.0001 versus 1.0e-05), `warmup_bias_lr` (0.005 versus 0.01) and the project name (`seg_training_result` versus `training_result_yolo11m-seg`); where they differ, `args.yaml` governs. The Comet ML login cell holds placeholder credentials and must be edited or removed before execution. |
| `inference_seg.ipynb` | `use_model/inference_seg.ipynb` | Single-image inference example with the trained weights. |

Weights (`best.pt`), the labelled dataset (image counts of the val and test splits are in `docs/SEG_XCHECK.md`; the training split is not counted here) and videos are not included; they are available from the corresponding author. The files in this folder are distributed under the Apache License 2.0 (`LICENSE` in this folder), unlike the rest of the repository, which is MIT.

Equivalent Ultralytics command lines (from `args.yaml`):

```powershell
# training (as archived; two GPUs, disk cache)
yolo segment train model=yolo11m-seg.pt data=dataset_seg/yolo_data_integrated/dataset.yaml imgsz=512 batch=32 epochs=100 patience=25 optimizer=AdamW lr0=0.005 lrf=0.0001 weight_decay=0.0007 warmup_epochs=3 cos_lr=True dropout=0.15 mosaic=0.8 mixup=0.5 cutmix=0.5 copy_paste=0.3 translate=0.15 scale=0.3 seed=42 deterministic=True device=0,1 workers=8 cache=disk project=training_result_yolo11m-seg name=yolo11m-seg_cow
# validation on a named split (what scripts/segmentation_xcheck.py does programmatically for val and test)
yolo segment val model=best.pt data=dataset.yaml imgsz=512 split=test batch=16 conf=0.001 iou=0.7 half=False plots=False
```

The comparison of both splits with the manuscript's Table 1 and Results values is in `docs/SEG_XCHECK.md`. Multi-object tracking (ByteTrack in the manuscript; `tracker: botsort.yaml` is only the Ultralytics default recorded in `args.yaml`) is not exercised by these notebooks and cannot be verified from this material.

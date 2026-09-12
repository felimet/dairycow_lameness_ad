# Segmentation cross-check

The manuscript reports three sets of metrics for the YOLOv11m-seg model. Table 1 gives the validation-split metrics (n = 1,101 images) logged by the training framework at the final training epoch. The Results add the logged validation metrics of the retained best-fitness checkpoint (epoch 96 of 100; used for silhouette extraction and included with the code deposit) and that checkpoint's evaluation on the 552-image held-out test split, which was produced by this script. Both splits are re-validated here with the retained checkpoint.

| Source | Split | n images | Type | P | R | mAP50 | mAP50-95 |
|---|---|---:|---|---:|---:|---:|---:|
| Manuscript Table 1 (val split, final epoch 100) | val | 1101 | mask | 0.992 | 0.994 | 0.995 | 0.839 |
| Manuscript Table 1 (val split, final epoch 100) | val | 1101 | box | NA | NA | 0.995 | 0.990 |
| Manuscript Results, retained checkpoint (epoch 96), logged val metrics | val | 1101 | mask | 0.993 | 0.996 | 0.995 | 0.841 |
| Manuscript Results, retained checkpoint (epoch 96), test split | test | 552 | mask | 0.989 | 0.992 | 0.995 | 0.850 |
| Manuscript Results, retained checkpoint (epoch 96), test split | test | 552 | box | NA | NA | 0.995 | 0.989 |
| This script, retained checkpoint | val | 1101 | mask | 0.993382 | 0.996233 | 0.994683 | 0.842821 |
| This script, retained checkpoint | val | 1101 | box | 0.993382 | 0.996233 | 0.994683 | 0.988367 |
| This script, retained checkpoint | test | 552 | mask | 0.989094 | 0.992391 | 0.994561 | 0.849982 |
| This script, retained checkpoint | test | 552 | box | 0.989094 | 0.992391 | 0.994561 | 0.989395 |

Agreement at the reported precision (3 decimals):

- Manuscript Table 1 (val split, final epoch 100): 2 of 6 reported values reproduced, mean absolute deviation 0.0016; differences mask precision 0.993 versus 0.992; mask recall 0.996 versus 0.994; mask map 0.843 versus 0.839; box map 0.988 versus 0.990.
- Manuscript Results, retained checkpoint (epoch 96), logged val metrics: 3 of 4 reported values reproduced, mean absolute deviation 0.0007; differences mask map 0.843 versus 0.841.
- Manuscript Results, retained checkpoint (epoch 96), test split: 6 of 6 reported values reproduced, mean absolute deviation 0.0003; all reported values reproduced.

Exact reproduction is expected only for the test-split row, since the Results report the evaluation this script produced. The val-split row is compared with two logged values of the training framework: the epoch-96 log of the same checkpoint, from which it differs only by the validator version and batching, and the epoch-100 log behind Table 1, from which it additionally differs by the epoch. The archived args.yaml records split=val for training-time validation and does not record the Ultralytics version.

Ultralytics 8.4.142; imgsz=512; batch=16; conf=0.001; iou=0.7; device=0; FP32; workers=0; no augmentation or plots.

```powershell
conda run -n cowlame python scripts/segmentation_xcheck.py --weights <path>/yolo11m-seg_cow/weights/best.pt --dataset <path>/dataset_seg/yolo_data_integrated/dataset.yaml --out results
conda run -n cowlame python scripts/segmentation_xcheck.py --render-only --out results   # re-render this file from the stored records
```

The original dataset YAML contains an obsolete absolute Linux path. A temporary repository-local YAML replaces only the data root with the current absolute parent directory. Source YAML and weights SHA-256 hashes match before and after validation.
Ultralytics label-cache read/write functions are redirected to cache/segmentation; cache=False disables image caching. A Python audit hook rejects file mutations outside this repository and the cowlame environment. No source YAML, labels, images or weights are edited.

Full precision and timings: results/segmentation_xcheck.json; the manuscript targets are in cowlame/manuscript_reference.json (block `segmentation`).

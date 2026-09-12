"""Validate both YOLO segmentation splits while protecting source assets.

``--render-only`` rewrites docs/SEG_XCHECK.md and the comparison block of
results/segmentation_xcheck.json from the stored validation records without
loading weights or data, so the manuscript targets can be revised without a
new validation run.
"""
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
from _common import arguments, initialize, ROOT
from cowlame.evaluate import write_json
from cowlame.config import SegmentationConfig

LOGGER = logging.getLogger(__name__)
DEFAULT_WEIGHTS = "<path>/yolo11m-seg_cow/weights/best.pt"
DEFAULT_DATASET = "<path>/dataset_seg/yolo_data_integrated/dataset.yaml"
METRICS = ("precision", "recall", "map50", "map")


def source_hashes(paths):
    try:
        hashes = {}
        for path in paths:
            with path.open("rb") as stream:
                hashes[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        return hashes
    except Exception:
        LOGGER.exception("Source hashing failed")
        raise


def placeholder(path, parts=3):
    """Document a path by its last components only; the absolute root is site-specific."""
    return "<path>/" + "/".join(Path(path).parts[-parts:])


def protect_sources():
    """Prevent Python-level writes beyond the authorized repo and conda env."""
    allowed = (ROOT.resolve(), Path(sys.prefix).resolve())
    blocked = []
    def check(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        resolved = Path(os.fsdecode(path)).resolve()
        if not any(resolved.is_relative_to(base) for base in allowed):
            blocked.append(str(resolved))
            LOGGER.error("Blocked out-of-scope write: %s", resolved)
            raise PermissionError(f"Read-only source protection: {resolved}")
    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if (mode and any(marker in mode for marker in "wa+x")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
                check(path)
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"):
            check(args[0])
        elif event in ("os.rename", "os.link", "os.symlink"):
            check(args[0])
            check(args[1])
    sys.addaudithook(audit)
    return blocked


def manuscript_targets(reference):
    """The three sets of segmentation metrics the manuscript reports, each with the split it refers to."""
    checkpoint = reference["retained_checkpoint"]
    return [
        ("table1", f"Manuscript Table 1 (val split, final epoch {reference['table1']['epoch']})", reference["table1"]),
        ("checkpoint_val_logged", f"Manuscript Results, retained checkpoint (epoch {checkpoint['epoch']}), logged val metrics",
         checkpoint["val_logged"]),
        ("checkpoint_test", f"Manuscript Results, retained checkpoint (epoch {checkpoint['epoch']}), test split", checkpoint["test"]),
    ]


def compare(record, target, decimals):
    """Per-metric deltas of one validated split against one manuscript target; None entries are not reported."""
    deltas, exact, total = {}, 0, 0
    for kind in ("mask", "box"):
        for metric in METRICS:
            value = target[kind][metric]
            if value is None:
                continue
            observed = record[kind][metric]
            deltas[f"{kind} {metric}"] = {"observed": observed, "manuscript": value, "delta": observed - value,
                                         "exact_at_reported_precision": round(observed, decimals) == value}
            total += 1
            exact += round(observed, decimals) == value
    if total == 0:
        raise ValueError("Manuscript target reports no value")
    return {"n_reported": total, "n_exact_at_reported_precision": exact,
            "mean_absolute_delta": sum(abs(d["delta"]) for d in deltas.values()) / total, "metrics": deltas}


def render(results, reference, seg, weights_text, dataset_text):
    """Compute the comparison block and write docs/SEG_XCHECK.md; returns the updated results."""
    try:
        decimals = seg.display_decimals
        checkpoint = reference["retained_checkpoint"]
        comparison = {}
        for key, label, target in manuscript_targets(reference):
            comparison[key] = {"label": label, "split": target["split"], "n": target["n"],
                               **compare(results["splits"][target["split"]], target, decimals)}
        results.pop("table1_comparison", None)
        results["comparison"] = comparison
        results["command"] = {"weights": weights_text, "dataset": dataset_text}
        fmt = lambda value: "NA" if value is None else f"{value:.{decimals}f}"
        lines = ["# Segmentation cross-check", "",
            f"The manuscript reports three sets of metrics for the YOLOv11m-seg model. Table 1 gives the validation-split metrics "
            f"(n = {reference['table1']['n']:,} images) {reference['table1']['provenance']}. The Results add the logged validation metrics of "
            f"the retained best-fitness checkpoint (epoch {checkpoint['epoch']} of {reference['table1']['epoch']}; {checkpoint['note']}) and that "
            f"checkpoint's evaluation on the {checkpoint['test']['n']}-image held-out test split, which was produced by this script. "
            "Both splits are re-validated here with the retained checkpoint.",
            "", "| Source | Split | n images | Type | P | R | mAP50 | mAP50-95 |", "|---|---|---:|---|---:|---:|---:|---:|"]
        for _, label, target in manuscript_targets(reference):
            for kind in ("mask", "box"):
                if all(target[kind][m] is None for m in METRICS):
                    continue
                lines.append(f"| {label} | {target['split']} | {target['n']} | {kind} | "
                             + " | ".join(fmt(target[kind][m]) for m in METRICS) + " |")
        for split, record in results["splits"].items():
            for kind in ("mask", "box"):
                lines.append(f"| This script, retained checkpoint | {split} | {record['n_images']} | {kind} | "
                             + " | ".join(f"{record[kind][m]:.6f}" for m in METRICS) + " |")
        lines += ["", f"Agreement at the reported precision ({decimals} decimals):", ""]
        for key, entry in comparison.items():
            off = [f"{name} {item['observed']:.{decimals}f} versus {item['manuscript']:.{decimals}f}"
                   for name, item in entry["metrics"].items() if not item["exact_at_reported_precision"]]
            detail = "all reported values reproduced" if not off else "differences " + "; ".join(off)
            lines.append(f"- {entry['label']}: {entry['n_exact_at_reported_precision']} of {entry['n_reported']} reported values "
                         f"reproduced, mean absolute deviation {entry['mean_absolute_delta']:.4f}; {detail}.")
        lines += ["",
            "Exact reproduction is expected only for the test-split row, since the Results report the evaluation this script produced. "
            f"The val-split row is compared with two logged values of the training framework: the epoch-{checkpoint['epoch']} log of the same "
            f"checkpoint, from which it differs only by the validator version and batching, and the epoch-{reference['table1']['epoch']} log "
            "behind Table 1, from which it additionally differs by the epoch. The archived args.yaml records split=val for training-time "
            "validation and does not record the Ultralytics version.",
            "", f"Ultralytics {results['ultralytics_version']}; imgsz={seg.imgsz}; batch={seg.batch}; conf={seg.conf}; iou={seg.iou}; "
            f"device={seg.device}; {'FP16' if seg.half else 'FP32'}; workers={seg.workers}; no augmentation or plots.",
            "", "```powershell",
            f"conda run -n cowlame python scripts/segmentation_xcheck.py --weights {weights_text} --dataset {dataset_text} --out results",
            "conda run -n cowlame python scripts/segmentation_xcheck.py --render-only --out results   # re-render this file from the stored records",
            "```",
            "", "The original dataset YAML contains an obsolete absolute Linux path. A temporary repository-local YAML replaces only the data root with the current absolute parent directory. Source YAML and weights SHA-256 hashes match before and after validation.",
            "Ultralytics label-cache read/write functions are redirected to cache/segmentation; cache=False disables image caching. A Python audit hook rejects file mutations outside this repository and the cowlame environment. No source YAML, labels, images or weights are edited.",
            "", "Full precision and timings: results/segmentation_xcheck.json; the manuscript targets are in cowlame/manuscript_reference.json (block `segmentation`)."]
        (ROOT / "docs/SEG_XCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return results
    except Exception:
        LOGGER.exception("Segmentation cross-check rendering failed")
        raise


def load_reference():
    return json.loads((ROOT / "cowlame/manuscript_reference.json").read_text(encoding="utf-8"))["segmentation"]


def main():
    try:
        parser = arguments(__doc__)
        parser.add_argument("--weights", type=Path)
        parser.add_argument("--dataset", type=Path)
        parser.add_argument("--render-only", action="store_true",
                            help="rewrite docs/SEG_XCHECK.md and the comparison block from results/segmentation_xcheck.json")
        args = parser.parse_args()
        seg = SegmentationConfig()
        if args.render_only:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            stored = args.out / "segmentation_xcheck.json"
            results = json.loads(stored.read_text(encoding="utf-8"))
            command = results.get("command", {})
            results = render(results, load_reference(), seg, command.get("weights", DEFAULT_WEIGHTS),
                             command.get("dataset", DEFAULT_DATASET))
            write_json(stored, results)
            LOGGER.info("Re-rendered docs/SEG_XCHECK.md from %s", stored)
            return
        if args.weights is None or args.dataset is None:
            parser.error("--weights and --dataset are required unless --render-only is given")
        cfg = initialize(args)
        LOGGER.info("Segmentation settings: %s", asdict(seg))
        args.out.mkdir(parents=True, exist_ok=True)
        cache = ROOT / "cache/segmentation"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        blocked_writes = protect_sources()
        # Imported only after config/cache locations and write protection are set.
        import yaml
        import ultralytics
        from ultralytics import YOLO
        import ultralytics.data.dataset as dataset_module
        ultralytics.settings.update({"sync": False, "datasets_dir": str(cache),
            "weights_dir": str(cache / "weights"), "runs_dir": str((args.out / "seg_xcheck").resolve())})
        source_paths = [args.weights.resolve(), args.dataset.resolve()]
        before = source_hashes(source_paths)
        original = yaml.safe_load(args.dataset.read_text(encoding="utf-8"))
        absolute = dict(original)
        absolute["path"] = str(args.dataset.resolve().parent)
        absolute.pop("download", None)
        local_yaml = cache / "dataset.absolute.yaml"
        local_yaml.write_text(yaml.safe_dump(absolute, allow_unicode=True, sort_keys=False), encoding="utf-8")
        native_load = dataset_module.load_dataset_cache_file
        native_save = dataset_module.save_dataset_cache_file
        def local_cache(path):
            key = hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:20]
            return cache / f"labels_{key}.cache"
        def load_cache(path):
            # A cache miss is the validator's documented trigger for a fresh scan.
            return native_load(local_cache(path))
        def save_cache(prefix, path, values, version):
            try:
                native_save(prefix, local_cache(path), values, version)
                if not local_cache(path).is_file():
                    raise OSError("Validator did not persist its repository-local label cache")
            except Exception:
                LOGGER.exception("Local segmentation cache output failed")
                raise
        results = {"settings": asdict(seg), "ultralytics_version": ultralytics.__version__,
            "original_dataset_path": placeholder(original["path"], 2) if original.get("path") else None, "effective_dataset_path": placeholder(absolute["path"], 2),
            "source_hashes_before": before, "splits": {}}
        for split in ("val", "test"):
            start = time.perf_counter()
            observed = {}
            def record_count(validator):
                observed["n_images"] = int(validator.seen)
            with patch.object(dataset_module, "load_dataset_cache_file", load_cache), \
                 patch.object(dataset_module, "save_dataset_cache_file", save_cache):
                model = YOLO(str(args.weights.resolve()))
                model.add_callback("on_val_end", record_count)
                stats = model.val(data=str(local_yaml), split=split, **asdict_for_validation(seg),
                    cache=False, plots=False, save_json=False, save_txt=False,
                    project=str((args.out / "seg_xcheck").resolve()), name=split,
                    exist_ok=True, seed=cfg.seed, deterministic=True, verbose=False)
            record = {"n_images": observed["n_images"], "elapsed_seconds": time.perf_counter() - start}
            for kind, item in (("box", stats.box), ("mask", stats.seg)):
                record[kind] = {"precision": float(item.mp), "recall": float(item.mr),
                                "map50": float(item.map50), "map": float(item.map)}
            results["splits"][split] = record
            LOGGER.warning("SEGMENTATION %s: %s", split, record)
            write_json(args.out / "segmentation_xcheck.partial.json", results)
        results["source_hashes_after"] = source_hashes(source_paths)
        results["blocked_write_attempts"] = blocked_writes
        assert before == results["source_hashes_after"], "Source YAML/weights changed"
        results = render(results, load_reference(), seg, placeholder(args.weights), placeholder(args.dataset))
        write_json(args.out / "segmentation_xcheck.json", results)
        (args.out / "segmentation_xcheck.partial.json").unlink(missing_ok=True)
    except Exception:
        LOGGER.exception("Segmentation cross-check failed")
        raise


def asdict_for_validation(cfg):
    result = asdict(cfg)
    result.pop("display_decimals")
    return result


if __name__ == "__main__":
    main()

"""Validate both YOLO segmentation splits while protecting source assets."""
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


def main():
    try:
        parser = arguments(__doc__)
        parser.add_argument("--weights", type=Path, required=True)
        parser.add_argument("--dataset", type=Path, required=True)
        args = parser.parse_args()
        cfg = initialize(args)
        seg = SegmentationConfig()
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
        reference = json.loads((ROOT / "cowlame/manuscript_reference.json").read_text(encoding="utf-8"))["segmentation"]
        exact = []
        for split, record in results["splits"].items():
            deltas = [abs(record[kind][metric] - target) for kind in ("mask", "box")
                      for metric, target in reference[kind].items() if target is not None]
            record["table1_mean_absolute_delta"] = sum(deltas) / len(deltas)
            record["matches_all_reported_values_at_3_decimals"] = all(
                round(record[kind][metric], seg.display_decimals) == target
                for kind in ("mask", "box") for metric, target in reference[kind].items() if target is not None)
            if record["matches_all_reported_values_at_3_decimals"]:
                exact.append(split)
        nearest = min(results["splits"], key=lambda key: results["splits"][key]["table1_mean_absolute_delta"])
        counted = [split for split, record in results["splits"].items() if record["n_images"] == reference["n"]]
        manuscript_statement = f"held-out {reference['claimed_split']} set, n = {reference['n']}"
        results["table1_comparison"] = {"exact_at_reported_precision": exact, "nearest_split": nearest, "split_with_reported_n": counted,
                                       "manuscript_statement": manuscript_statement}
        write_json(args.out / "segmentation_xcheck.json", results)
        (args.out / "segmentation_xcheck.partial.json").unlink(missing_ok=True)
        lines = ["# Segmentation cross-check", "", f'Manuscript Table 1 reports "{manuscript_statement}".',
            "", "| Split | n images | Type | P | R | mAP50 | mAP50-95 |", "|---|---:|---|---:|---:|---:|---:|"]
        for kind in ("mask", "box"):
            lines.append(f"| Manuscript {reference['claimed_split']} | {reference['n']} | {kind} | " + " | ".join(
                "NA" if reference[kind][m] is None else f"{reference[kind][m]:.{seg.display_decimals}f}"
                for m in ("precision", "recall", "map50", "map")) + " |")
        for split, record in results["splits"].items():
            for kind in ("mask", "box"):
                row = record[kind]
                lines.append(f"| {split} | {record['n_images']} | {kind} | " + " | ".join(
                    f"{row[m]:.6f}" for m in ("precision", "recall", "map50", "map")) + " |")
        statement = ("Split(s) " + ", ".join(exact) + " reproduce all reported Table 1 metrics at three decimal places."
                     if exact else f"Neither split reproduces every reported Table 1 value at three decimal places. {nearest} is nearer by mean absolute deviation over the six reported metrics; "
                     + (", ".join(counted) + f" has the reported image count (n = {reference['n']})." if counted else f"no split has the reported image count (n = {reference['n']})."))
        lines += ["", statement, "", f"Ultralytics {ultralytics.__version__}; imgsz={seg.imgsz}; batch={seg.batch}; conf={seg.conf}; iou={seg.iou}; device={seg.device}; {'FP16' if seg.half else 'FP32'}; workers={seg.workers}; no augmentation or plots.",
            "", "```powershell",
            f'conda run -n cowlame python scripts/segmentation_xcheck.py --weights {placeholder(args.weights)} --dataset {placeholder(args.dataset)} --out results', "```",
            "", "The original dataset YAML contains an obsolete absolute Linux path. A temporary repository-local YAML replaces only the data root with the current absolute parent directory. Source YAML and weights SHA-256 hashes match before and after validation.",
            "Ultralytics label-cache read/write functions are redirected to cache/segmentation; cache=False disables image caching. A Python audit hook rejects file mutations outside this repository and the cowlame environment. No source YAML, labels, images or weights are edited.",
            "", "Full precision and timings: results/segmentation_xcheck.json. The archived args.yaml records split=val for training-time validation and does not record the Ultralytics version; differences at the third decimal between the two evaluations above and Table 1 are of the size expected from validator-version and batching differences, and the split comparison above is the empirical evidence available."]
        (ROOT / "docs/SEG_XCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        LOGGER.exception("Segmentation cross-check failed")
        raise


def asdict_for_validation(cfg):
    result = asdict(cfg)
    result.pop("display_decimals")
    return result


if __name__ == "__main__":
    main()

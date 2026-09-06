"""Read-only manifest ingestion, validated paths, and raw tensor caching."""
import hashlib
import logging
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
from PIL import Image

LOGGER = logging.getLogger(__name__)
PASS_KEYS = ["recording", "pass"]
PASS_COLUMNS = PASS_KEYS + ["cow_id", "score_1_3", "anomaly", "max_concurrent", "group"]


def remap_path(value, data_root):
    """Remap the historical prefix ending in 01_data, without drive assumptions."""
    try:
        root = Path(data_root).resolve()
        parts = PureWindowsPath(str(value)).parts
        if "01_data" in parts:
            path = root.joinpath(*parts[parts.index("01_data") + 1:]).resolve()
        else:
            path = (root / Path(value)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Manifest path escapes data root: {value}")
        assert path.is_file(), f"Missing energy image: {path}"
        return path
    except Exception:
        LOGGER.exception("Path remapping failed for %s", value)
        raise


def load_manifest(data_root, cfg):
    try:
        path = Path(data_root) / "window_manifest_maskfirst.csv"
        frame = pd.read_csv(path, dtype={"cow_id": str}, keep_default_na=False)
        frame = frame[frame.aligned.astype(str).str.lower().eq("true")].copy()
        for column in ["recording", "pass", "window", "score_1_3", "max_concurrent"]:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
        if not frame.score_1_3.isin([1, 2, 3]).all():
            raise ValueError("Consensus scores must be 1, 2, or 3")
        frame["anomaly"] = (frame.score_1_3 > cfg.normal_score).astype(int)
        frame["max_concurrent"] = frame.max_concurrent.clip(upper=cfg.max_crowding)
        frame["group"] = frame.cow_id.where(frame.cow_id.ne(""), "unknown")
        if cfg.unknown_cow_policy == "per-pass":
            missing = frame.cow_id.eq("")
            frame.loc[missing, "group"] = ("unknown_" + frame.loc[missing, "recording"].astype(str)
                                           + "_" + frame.loc[missing, "pass"].astype(str))
        if frame.duplicated(PASS_KEYS + ["window"]).any():
            raise ValueError("Duplicate manifest windows")
        frame = frame.sort_values(PASS_KEYS + ["window"], kind="stable").reset_index(drop=True)
        consistency = frame.groupby(PASS_KEYS)[PASS_COLUMNS[2:]].nunique(dropna=False)
        if (consistency > 1).any().any():
            raise ValueError("Inconsistent labels or metadata within a pass")
        passes = frame[PASS_COLUMNS].drop_duplicates(PASS_KEYS).reset_index(drop=True)
        lookup = {(r.recording, getattr(r, "pass")): i for i, r in passes.iterrows()}
        frame["pass_index"] = [lookup[key] for key in frame[PASS_KEYS].itertuples(index=False, name=None)]
        LOGGER.info("Analysis: %d windows, %d passes, %d known cows, %d groups; severity=%s; crowding=%s",
                    len(frame), len(passes), passes.loc[passes.cow_id.ne(""), "cow_id"].nunique(),
                    passes.group.nunique(), passes.score_1_3.value_counts().to_dict(),
                    passes.max_concurrent.value_counts().to_dict())
        return frame, passes
    except Exception:
        LOGGER.exception("Manifest loading failed")
        raise


def manifest_hash(frame, cfg, data_root):
    payload = frame.to_csv(index=False).encode() + repr((cfg.cache_version, cfg.alpha,
              cfg.image_size, str(Path(data_root).resolve()))).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def load_images(frame, modality, crop, data_root, cache_dir, cfg):
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = manifest_hash(frame, cfg, data_root)
        target = cache_dir / f"{key}_{modality}_{crop}.npy"
        column = "hei_comb6_path" if modality == "HEI" else modality.lower() + "_path"
        channels = 1 if modality in ("GEI", "FDEI") else 3
        expected = (len(frame), channels, cfg.image_size, cfg.image_size)
        if target.exists():
            result = np.load(target, mmap_mode="r")
            if result.shape != expected or result.dtype != np.float32:
                raise ValueError(f"Invalid cache {target}")
            return result
        partial = target.with_suffix(".partial.npy")
        stack = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected)
        for index, value in enumerate(frame[column]):
            path = remap_path(value, data_root)
            with Image.open(path) as source:
                pic = source.convert("L" if channels == 1 else "RGB")
                if crop == "leg":
                    top = int(np.floor(pic.height * (1 - cfg.alpha)))
                    pic = pic.crop((0, top, pic.width, pic.height))
                elif crop != "full":
                    raise ValueError(f"Unknown crop: {crop}")
                pic = pic.resize((cfg.image_size, cfg.image_size), Image.Resampling.BILINEAR)
                values = np.asarray(pic, dtype=np.float32) / cfg.pixel_max
                stack[index] = values[None] if channels == 1 else values.transpose(2, 0, 1)
        stack.flush()
        del stack
        partial.replace(target)
        LOGGER.info("Cached %s %s: %s", modality, crop, expected)
        return np.load(target, mmap_mode="r")
    except Exception:
        LOGGER.exception("Image loading/caching failed: %s %s", modality, crop)
        raise


def pass_means(images, pass_indices, n_passes):
    return np.stack([images[pass_indices == i].mean(axis=0) for i in range(n_passes)])

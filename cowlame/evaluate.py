"""Cow-level folds, normal-only fitting and pass-level out-of-fold scores."""
from dataclasses import asdict
import gc
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold
import torch

from .data import pass_means, load_images, manifest_hash
from .detectors import DETECTORS

LOGGER = logging.getLogger(__name__)
CONFIG_KEYS = ["detector", "modality", "crop"]


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        raise ValueError("Metrics require two classes and finite scores")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) != 0)
    best = int(np.argmax(f1[:-1]))
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores)),
            "max_f1": float(f1[best]), "threshold": float(thresholds[best])}


def folds(passes, cfg):
    splitter = GroupKFold(n_splits=cfg.folds)
    for fold, (train, test) in enumerate(splitter.split(passes, groups=passes.group)):
        assert set(passes.iloc[train].group).isdisjoint(passes.iloc[test].group), "Cow leakage"
        LOGGER.info("Fold %d: train=%d test=%d normal_train=%d", fold, len(train), len(test),
                    int(passes.iloc[train].anomaly.eq(0).sum()))
        yield fold, train, test


def aggregate(scores, indices, pass_ids):
    return np.asarray([np.mean(scores[indices == i], dtype=np.float64) for i in pass_ids])


def summarize(oof, cfg):
    per_fold = [dict(fold=int(fold), **metrics(group.anomaly, group.score))
                for fold, group in oof.groupby("fold")]
    result = {"n_passes": len(oof), "folds": per_fold, "pooled": metrics(oof.anomaly, oof.score)}
    for metric in ("auroc", "auprc", "max_f1"):
        values = np.array([row[metric] for row in per_fold])
        half = t.ppf((1 + cfg.confidence) / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
        result[metric] = {"mean": float(values.mean()), "ci_low": float(values.mean() - half),
                          "ci_high": float(values.mean() + half)}
    return result


def cross_validate(images, window_pass, passes, detector, cfg, features=None, curve_callback=None):
    try:
        oof = passes.copy()
        oof["score"] = np.nan
        oof["fold"] = -1
        means = pass_means(images, window_pass, len(passes)) if detector == "subspace" else None
        for fold, train_ids, test_ids in folds(passes, cfg):
            normal_ids = train_ids[passes.iloc[train_ids].anomaly.to_numpy() == 0]
            train_windows = np.isin(window_pass, normal_ids)
            test_windows = np.isin(window_pass, test_ids)
            assert not passes.iloc[normal_ids].anomaly.any()
            model = DETECTORS[detector](cfg, channels=images.shape[1])
            if detector == "subspace":
                model.fit(means[normal_ids])
                scores = model.score(means[test_ids])
            elif detector == "patchcore":
                if features is None:
                    features = model.features(images)
                model.fit_features(features[train_windows])
                window_scores = model.score_features(features[test_windows])
                scores = aggregate(window_scores, window_pass[test_windows], test_ids)
            else:
                test_x = images[test_windows]
                def epoch_callback(epoch, loss):
                    pass_scores = aggregate(model.score(test_x), window_pass[test_windows], test_ids)
                    auc = metrics(passes.iloc[test_ids].anomaly, pass_scores)["auroc"]
                    if curve_callback:
                        curve_callback({"fold": fold, "epoch": epoch, "train_loss": loss,
                                        "heldout_auroc": auc})
                model.fit(images[train_windows], epoch_callback)
                scores = aggregate(model.score(test_x), window_pass[test_windows], test_ids)
                del test_x
            oof.loc[test_ids, "score"] = scores
            oof.loc[test_ids, "fold"] = fold
            LOGGER.info("%s fold %d metrics: %s", detector, fold,
                        metrics(passes.iloc[test_ids].anomaly, scores))
            del model
            gc.collect()
            if cfg.device == "cuda":
                torch.cuda.empty_cache()
        assert oof.score.notna().all() and (oof.fold >= 0).all(), "Incomplete OOF coverage"
        return oof
    except Exception:
        LOGGER.exception("Cross-validation failed for %s", detector)
        raise


def write_json(path, content):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    except Exception:
        LOGGER.exception("JSON output failed: %s", path)
        raise


def collect_runs(out):
    try:
        out = Path(out)
        files = sorted((out / "runs").glob("*_oof.csv"))
        if files:
            pd.concat([pd.read_csv(p, dtype={"cow_id": str}, keep_default_na=False) for p in files],
                      ignore_index=True).to_csv(out / "oof_scores.csv", index=False)
        for detector in ("convae", "stfpm"):
            curves = sorted((out / "runs").glob(f"{detector}_*_curves.csv"))
            if curves:
                pd.concat([pd.read_csv(p) for p in curves], ignore_index=True).to_csv(
                    out / f"{detector}_curves.csv", index=False)
    except Exception:
        LOGGER.exception("Collecting run outputs failed")
        raise


def run_configuration(frame, passes, detector, modality, crop, data_root, out, cache, cfg,
                      images=None, force=False, features=None):
    try:
        start = time.perf_counter()
        folder = Path(out) / "runs"
        folder.mkdir(parents=True, exist_ok=True)
        key = f"{detector}_{modality}_{crop}"
        score_file = folder / f"{key}_oof.csv"
        meta_file = folder / f"{key}.json"
        signature = json.loads(json.dumps({"config": asdict(cfg),
                    "manifest_hash": manifest_hash(frame, cfg, data_root)}))
        if score_file.exists() and meta_file.exists() and not force:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            if metadata["signature"] != signature:
                raise ValueError(f"Existing {key} uses different settings; use a different --out or --force")
            LOGGER.info("Reusing complete run %s", key)
            return pd.read_csv(score_file, dtype={"cow_id": str}, keep_default_na=False), metadata["summary"]
        if images is None:
            images = load_images(frame, modality, crop, data_root, cache, cfg)
        curve_file = folder / f"{key}_curves.csv"
        if curve_file.exists():
            curve_file.unlink()
        def save_curve(row):
            try:
                row.update(detector=detector, modality=modality, crop=crop)
                pd.DataFrame([row]).to_csv(curve_file, mode="a", header=not curve_file.exists(), index=False)
            except Exception:
                LOGGER.exception("Epoch curve output failed")
                raise
        oof = cross_validate(images, frame.pass_index.to_numpy(), passes, detector, cfg,
                             features=features, curve_callback=save_curve)
        for column, value in zip(CONFIG_KEYS, (detector, modality, crop)):
            oof[column] = value
        oof = oof[CONFIG_KEYS + [c for c in oof.columns if c not in CONFIG_KEYS]]
        result = summarize(oof, cfg)
        partial = score_file.with_suffix(".partial.csv")
        oof.to_csv(partial, index=False)
        partial.replace(score_file)
        write_json(meta_file, {"signature": signature, "summary": result,
                              "elapsed_seconds": time.perf_counter() - start})
        collect_runs(out)
        LOGGER.warning("HEADLINE %s: AUROC %.6f [%.6f, %.6f]; pooled %.6f; AUPRC %.6f; maxF1 %.6f",
                       key, result["auroc"]["mean"], result["auroc"]["ci_low"],
                       result["auroc"]["ci_high"], result["pooled"]["auroc"],
                       result["auprc"]["mean"], result["max_f1"]["mean"])
        return oof, result
    except Exception:
        LOGGER.exception("Configuration failed: %s %s %s", detector, modality, crop)
        raise

"""Cohort counts from the aligned manifest, cross-checked against the pass-level expert-score file."""
from dataclasses import replace
import hashlib
import logging
from pathlib import Path
import pandas as pd
from _common import arguments
from cowlame.config import Config
from cowlame.data import load_manifest, PASS_KEYS
from cowlame.evaluate import folds, write_json

LOGGER = logging.getLogger(__name__)
MANIFEST = "window_manifest_maskfirst.csv"
GROUND_TRUTH = "gt_passes_001_118.csv"
EXPERT_COLUMNS = ("A", "B", "C")
UNUSED_COLUMNS = ("D",)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    try:
        args = arguments(__doc__).parse_args()
        logging.basicConfig(level=logging.INFO)
        cfg = replace(Config(), unknown_cow_policy=args.unknown_cow_policy)
        LOGGER.info("Cohort settings: %s", cfg)
        raw = pd.read_csv(args.data_root / MANIFEST, dtype=str, keep_default_na=False)
        frame, passes = load_manifest(args.data_root, cfg)
        truth = pd.read_csv(args.data_root / GROUND_TRUTH, dtype=str, keep_default_na=False)
        missing = [c for c in ("video_name", "final", *EXPERT_COLUMNS, *UNUSED_COLUMNS) if c not in truth.columns]
        if missing:
            raise ValueError(f"{GROUND_TRUTH} lacks columns {missing}")
        if truth.video_name.duplicated().any():
            raise ValueError(f"Duplicate video_name in {GROUND_TRUTH}")
        aligned = frame.drop_duplicates(PASS_KEYS)[PASS_KEYS + ["video_name", "score_1_3"]]
        merged = aligned.merge(truth[["video_name", "final"]], on="video_name", how="left", validate="one_to_one")
        final = pd.to_numeric(merged.final, errors="coerce")
        agreement = final == merged.score_1_3          # NaN (pass absent from the file) compares False
        if not agreement.all():
            LOGGER.warning("Consensus mismatch on %d passes:\n%s", int((~agreement).sum()),
                           merged.loc[~agreement].to_string())
        fold_rows = [{"fold": fold, "train_passes": len(train), "test_passes": len(test),
                      "normal_train_passes": int(passes.iloc[train].anomaly.eq(0).sum())}
                     for fold, train, test in folds(passes, cfg)]
        unknown = passes.loc[passes.cow_id.eq(""), PASS_KEYS + ["score_1_3"]]
        cohort = {
            "raw_manifest_windows": len(raw),
            "raw_manifest_passes": int(raw.drop_duplicates(PASS_KEYS).shape[0]),
            "aligned_windows": len(frame),
            "aligned_passes": len(passes),
            "known_cows": int(passes.loc[passes.cow_id.ne(""), "cow_id"].nunique()),
            "groups_with_unknown": int(passes.group.nunique()),
            "unknown_passes": [{"recording": int(recording), "pass": int(pass_id), "score_1_3": int(score)}
                               for recording, pass_id, score in unknown.itertuples(index=False, name=None)],
            "consensus_counts": {str(k): int(v) for k, v in passes.score_1_3.value_counts().items()},
            "crowding_counts": {str(k): int(v) for k, v in passes.max_concurrent.value_counts().items()},
            "consensus_matches_gt_on_every_aligned_pass": bool(agreement.all()),
            "folds": fold_rows,
            "source_sha256": {MANIFEST: sha256(args.data_root / MANIFEST),
                              GROUND_TRUTH: sha256(args.data_root / GROUND_TRUTH)},
            "expert_columns": list(EXPERT_COLUMNS),
            "unused_columns": list(UNUSED_COLUMNS),
        }
        write_json(args.out / "cohort.json", cohort)
        LOGGER.info("Cohort: %s", {k: v for k, v in cohort.items() if k not in ("folds", "unknown_passes")})
    except Exception:
        LOGGER.exception("Cohort export failed")
        raise


if __name__ == "__main__":
    main()

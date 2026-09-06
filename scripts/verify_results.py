"""Acceptance check for the completed deposited artifact set (no retraining)."""
import json
import logging
import pandas as pd
from _common import arguments, ROOT
from cowlame.config import Config
from cowlame.evaluate import CONFIG_KEYS, metrics


def main():
    try:
        args = arguments(__doc__).parse_args()
        logging.basicConfig(level=logging.INFO)
        cfg = Config()
        logging.info("Acceptance settings: %s", cfg)
        required = ["REPORT.md", "summary.json", "oof_scores.csv", "convae_curves.csv", "stfpm_curves.csv",
                    "table2_legcrop.csv", "crowding.csv", "permutation.json", "severity.json", "hei_perm.csv",
                    "fusion_percow.json", "fig4_bars.csv", "fig5_crowding.csv", "ei_xcheck.csv",
                    "segmentation_xcheck.json", "runtime_main.json", "runtime_ablations.json", "cohort.json", "window_lists.json"]
        for name in required:
            assert (args.out / name).is_file(), f"Missing result: {name}"
        for name in ["README.md", "LICENSE", "CITATION.cff", ".zenodo.json", "environment.yml", "requirements.txt",
                     "docs/IMPLEMENTATION_NOTES.md", "results/RUNTIME.md", "docs/EI_XCHECK.md", "docs/SEG_XCHECK.md",
                     "upstream/segmentation/args.yaml", "upstream/segmentation/LICENSE", "upstream/segmentation/train_yolo11_seg.ipynb",
                     "upstream/segmentation/inference_seg.ipynb"]:
            assert (ROOT / name).is_file(), f"Missing deliverable: {name}"
        oof = pd.read_csv(args.out / "oof_scores.csv", dtype={"cow_id": str}, keep_default_na=False)
        expected = {(d, m, "leg") for d in cfg.detectors for m in cfg.modalities}
        expected |= {(d, m, "full") for d in ("patchcore", "subspace") for m in cfg.modalities}
        expected |= {("patchcore", f"HEI_comb{i}", "leg") for i in range(1, 7)}
        expected |= {("subspace", "GEI_CGI_FDEI", "leg"), ("patchcore", "CGI_percow", "leg")}
        expected |= {("patchcore", "GEI_CGI_FDEI", "leg")}
        assert set(oof.groupby(CONFIG_KEYS).groups) == expected, "Configuration grid is incomplete"
        assert not oof.duplicated(CONFIG_KEYS + ["recording", "pass"]).any()
        for key, group in oof.groupby(CONFIG_KEYS):
            assert len(group) == 252, (key, "unexpected pass count")
            assert group.score_1_3.value_counts().to_dict() == {2: 126, 1: 120, 3: 6}
            assert group.max_concurrent.value_counts().to_dict() == {3: 165, 2: 60, 1: 27}
            assert (group.groupby("group").fold.nunique() == 1).all(), (key, "cow leakage")
            assert set(group.fold) == set(range(cfg.folds))
            assert (group.anomaly == (group.score_1_3 >= 2)).all()
            metrics(group.anomaly, group.score)
        for detector in ("convae", "stfpm"):
            curves = pd.read_csv(args.out / f"{detector}_curves.csv")
            assert len(curves) == len(cfg.modalities) * cfg.folds * cfg.epochs
            for _, group in curves.groupby(["modality", "fold"]):
                assert set(group.epoch) == set(range(1, cfg.epochs + 1))
                assert group.heldout_auroc.between(0, 1).all() and (group.train_loss >= 0).all()
        assert len(pd.read_csv(args.out / "table2_legcrop.csv")) == 8
        assert len(pd.read_csv(args.out / "hei_perm.csv")) == 6
        for result in json.loads((args.out / "permutation.json").read_text(encoding="utf-8")).values():
            assert result["B"] == cfg.permutation_count
            assert result["p"] == result["exceedances"] / result["B"]
        cohort = json.loads((args.out / "cohort.json").read_text(encoding="utf-8"))
        sample = oof[(oof.detector == "patchcore") & (oof.modality == "CGI") & (oof.crop == "leg")]
        assert cohort["aligned_passes"] == len(sample) == sum(f["test_passes"] for f in cohort["folds"])
        assert [f["test_passes"] for f in cohort["folds"]] == sample.fold.value_counts().sort_index().tolist()
        assert {int(k): v for k, v in cohort["consensus_counts"].items()} == sample.score_1_3.value_counts().to_dict()
        assert {int(k): v for k, v in cohort["crowding_counts"].items()} == sample.max_concurrent.value_counts().to_dict()
        assert cohort["groups_with_unknown"] == sample.group.nunique()
        assert cohort["known_cows"] == sample.loc[sample.cow_id != "", "cow_id"].nunique()
        assert cohort["consensus_matches_gt_on_every_aligned_pass"] is True
        segmentation = json.loads((args.out / "segmentation_xcheck.json").read_text(encoding="utf-8"))
        assert set(segmentation["splits"]) == {"val", "test"}
        assert segmentation["splits"]["val"]["n_images"] == 1101
        assert segmentation["splits"]["test"]["n_images"] == 552
        assert segmentation["source_hashes_before"] == segmentation["source_hashes_after"]
        print(f"PASS: {len(expected)} configurations, {len(oof)} OOF rows, complete 40-epoch curves, cohort counts, energy-image and segmentation cross-check outputs, and deposit files")
    except Exception:
        logging.exception("Artifact acceptance failed")
        raise


if __name__ == "__main__":
    main()

"""Full/leg table, six HEI permutations, Subspace and PatchCore late fusion, per-cow normalization."""
import logging
import time
import numpy as np
import pandas as pd
from _common import arguments, initialize
from cowlame.data import load_manifest, load_images, pass_means
from cowlame.evaluate import run_configuration, summarize, write_json, collect_runs
from cowlame.stats import per_cow_normalize
from cowlame.detectors import PatchCore

LOGGER = logging.getLogger(__name__)


def main():
    try:
        args = arguments(__doc__).parse_args()
        cfg = initialize(args)
        start = time.perf_counter()
        frame, passes = load_manifest(args.data_root, cfg)
        table = []
        for detector in ("patchcore", "subspace"):
            for modality in cfg.modalities:
                row = {"detector": detector, "modality": modality}
                for crop in ("full", "leg"):
                    _, summary = run_configuration(frame, passes, detector, modality, crop,
                        args.data_root, args.out, args.cache, cfg, force=args.force)
                    row[crop + "_auroc"] = summary["auroc"]["mean"]
                    row[crop + "_pooled_auroc"] = summary["pooled"]["auroc"]
                # *_auroc/delta are fold means; *_pooled_auroc/pooled_delta are the pooled
                # out-of-fold estimand the manuscript's Table 2 reports.
                row["delta"] = row["leg_auroc"] - row["full_auroc"]
                row["pooled_delta"] = row["leg_pooled_auroc"] - row["full_pooled_auroc"]
                table.append(row)
        pd.DataFrame(table).to_csv(args.out / "table2_legcrop.csv", index=False)
        permutations = []
        for index in range(1, 7):
            _, summary = run_configuration(frame, passes, "patchcore", f"HEI_comb{index}", "leg",
                args.data_root, args.out, args.cache, cfg, force=args.force)
            permutations.append({"permutation": index, **summary["auroc"],
                                 "pooled_auroc": summary["pooled"]["auroc"]})
        pd.DataFrame(permutations).to_csv(args.out / "hei_perm.csv", index=False)
        # Average each modality first to keep fusion memory bounded; channel concatenation
        # is exactly concatenation of the flattened pass-mean vectors for this PCA model.
        means = [pass_means(load_images(frame, m, "leg", args.data_root, args.cache, cfg),
                            frame.pass_index.to_numpy(), len(passes)) for m in ("GEI", "CGI", "FDEI")]
        joined = np.concatenate(means, axis=1)
        pass_frame = frame.drop_duplicates(["recording", "pass"]).copy().reset_index(drop=True)
        pass_frame["pass_index"] = np.arange(len(passes))
        _, fusion = run_configuration(pass_frame, passes, "subspace", "GEI_CGI_FDEI", "leg",
            args.data_root, args.out, args.cache, cfg, images=joined, force=args.force)
        extractor = PatchCore(cfg)
        patch_features = np.concatenate([extractor.features(load_images(frame, modality, "leg",
            args.data_root, args.cache, cfg)) for modality in ("GEI", "CGI", "FDEI")], axis=1)
        _, patch_fusion = run_configuration(frame, passes, "patchcore", "GEI_CGI_FDEI", "leg",
            args.data_root, args.out, args.cache, cfg,
            images=load_images(frame, "GEI", "leg", args.data_root, args.cache, cfg),
            features=patch_features, force=args.force)
        del extractor, patch_features
        base, _ = run_configuration(frame, passes, "patchcore", "CGI", "leg",
            args.data_root, args.out, args.cache, cfg)
        normalized = per_cow_normalize(base)
        normalized["modality"] = "CGI_percow"
        normalized.to_csv(args.out / "percow_scores.csv", index=False)
        normalized.to_csv(args.out / "runs/patchcore_CGI_percow_leg_oof.csv", index=False)
        write_json(args.out / "fusion_percow.json", {"subspace_fusion": fusion,
            "percow": summarize(normalized, cfg), "percow_method": "OOF within-cow z-score, ddof=0; singletons raw",
            "patchcore_fusion": patch_fusion})
        write_json(args.out / "runtime_ablations.json", {"elapsed_seconds": time.perf_counter() - start})
        collect_runs(args.out)
    except Exception:
        LOGGER.exception("Ablation analysis failed")
        raise


if __name__ == "__main__":
    main()

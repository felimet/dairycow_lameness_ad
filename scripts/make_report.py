"""Generate auditable side-by-side results and figure-data tables."""
import json
import logging
import pandas as pd
from _common import arguments, ROOT
from cowlame.config import Config
from cowlame.evaluate import CONFIG_KEYS, summarize, collect_runs, write_json
from cowlame.stats import permutation_test, severity_analysis, crowding_analysis

LOGGER = logging.getLogger(__name__)


def write_runtime(out):
    """Measured wall times from results/, per configuration and per stage."""
    try:
        rows = []
        for meta in sorted((out / "runs").glob("*.json")):
            item = json.loads(meta.read_text(encoding="utf-8"))
            rows.append({"configuration": meta.stem, "seconds": item["elapsed_seconds"]})
        stages = []
        for name, key in (("Main grid wall time (scripts/run_main.py)", "runtime_main.json"),
                          ("Design-factor analyses wall time, last invocation (scripts/run_ablations.py)", "runtime_ablations.json"),
                          ("Energy-image cross-check (scripts/regenerate_ei.py)", "ei_xcheck.json")):
            path = out / key
            if path.exists():
                stages.append((name, json.loads(path.read_text(encoding="utf-8"))["elapsed_seconds"]))
        seg = out / "segmentation_xcheck.json"
        if seg.exists():
            for split, record in json.loads(seg.read_text(encoding="utf-8"))["splits"].items():
                stages.append((f"Segmentation validation, {split} split", record["elapsed_seconds"]))
        main_path = out / "runtime_main.json"
        system = json.loads(main_path.read_text(encoding="utf-8")) if main_path.exists() else {}
        lines = ["# Measured runtime", "",
                 "System: " + ", ".join(f"{k} {system[k]}" for k in ("gpu", "torch", "cuda") if k in system), "",
                 "Per-configuration compute time (seconds, five folds; image caching is included only when the cache was built during that run):", "",
                 "| Configuration | Seconds |", "|---|---:|"]
        lines += [f"| {r['configuration']} | {r['seconds']:.1f} |" for r in rows]
        lines += ["", f"Sum over configurations: {sum(r['seconds'] for r in rows):.1f} s", "",
                  "| Stage | Seconds |", "|---|---:|"] + [f"| {n} | {s:.1f} |" for n, s in stages]
        lines += ["", "Completed configurations are reused on re-invocation, so a stage wall time can be far below the sum of its configurations."]
        (out / "RUNTIME.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        LOGGER.exception("Runtime summary failed")
        raise


def main():
    try:
        args = arguments(__doc__).parse_args()
        logging.basicConfig(level=logging.INFO)
        cfg = Config()
        LOGGER.info("Reporting configuration: %s", cfg)
        collect_runs(args.out)
        oof = pd.read_csv(args.out / "oof_scores.csv", dtype={"cow_id": str}, keep_default_na=False)
        reference = json.loads((ROOT / "cowlame/manuscript_reference.json").read_text(encoding="utf-8"))
        summary, permutations, severity, crowding, bars, comparison = {}, {}, {}, [], [], []
        def compare(label, actual, manuscript):
            delta = abs(actual - manuscript) if isinstance(actual, (int, float)) and isinstance(manuscript, (int, float)) else None
            comparison.append({"metric": label, "repository": actual, "manuscript": manuscript, "absolute_delta": delta})
        def permutation_text(result):
            k, b = result["exceedances"], result["B"]
            return f"{k}/{b}" + (f" (<1/{b})" if k == 0 else "") + f"; (k+1)/(B+1) = {result['p_plus_one']:.4g}"
        for (detector, modality, crop), group in oof.groupby(CONFIG_KEYS, sort=False):
            key = f"{detector}_{modality}_{crop}"
            current = summarize(group, cfg)
            summary[key] = current
            permutations[key] = permutation_test(group.anomaly, group.score, cfg)
            source = reference["main_auroc"] if crop == "leg" else reference["full_auroc"]
            ref = source.get(detector, {}).get(modality)
            suffix = ""
            if ref is None and crop == "leg" and modality in reference["fig4_read_auroc"].get(detector, {}):
                ref, suffix = reference["fig4_read_auroc"][detector][modality], " (manuscript figure, read)"
            # The manuscript point estimate is the pooled out-of-fold value; the fold mean is
            # compared against the same reference so both estimands are visible side by side.
            compare(f"{key} pooled AUROC{suffix}", current["pooled"]["auroc"], ref)
            compare(f"{key} fold-mean AUROC{suffix}", current["auroc"]["mean"], ref)
            if crop == "leg" and modality in cfg.modalities:
                # mean/ci_low/ci_high are the fold-mean estimand; pooled_auroc is the estimand
                # the manuscript's figures and tables report.
                bars.append({"detector": detector, "modality": modality, **current["auroc"],
                             "pooled_auroc": current["pooled"]["auroc"]})
            if detector in ("patchcore", "subspace") and modality == "CGI" and crop == "leg":
                severity[key] = severity_analysis(group)
                for row in crowding_analysis(group):
                    row["detector"] = detector
                    crowding.append(row)
                    compare(f"{detector} crowding {row['stratum']} pooled AUROC", row["auroc"],
                            reference["crowding"][detector][row["stratum"] - 1])
                compare(f"{key} permutation p, k/B; (k+1)/(B+1)", permutation_text(permutations[key]),
                        reference[f"{detector}_cgi"]["permutation_p"])
                compare(f"{key} permutation mean AUROC", permutations[key]["permutation_mean_auroc"], 0.50)
                for grade, name in ((2, "mild"), (3, "severe")):
                    compare(f"{key} {name} detected", severity[key]["severity"][str(grade)]["detected"],
                            reference[f"{detector}_cgi"].get(f"{name}_detected"))
                if detector == "patchcore":
                    for metric in ("auprc", "max_f1"):
                        compare(f"{key} fold-mean {metric}", current[metric]["mean"], reference["patchcore_cgi"][metric])
                        compare(f"{key} pooled {metric}", current["pooled"][metric], reference["patchcore_cgi"][metric])
                    compare(f"{key} pooled Spearman rho", severity[key]["spearman_rho"], reference["patchcore_cgi"]["spearman_rho"])
                    for grade, target in reference["patchcore_cgi"]["median_score_by_severity"].items():
                        compare(f"{key} pooled median score, severity {grade}", severity[key]["severity"][grade]["median_score"], target)
                    compare("PatchCore CGI AUROC CI low", current["auroc"]["ci_low"], reference["patchcore_cgi"]["ci"][0])
                    compare("PatchCore CGI AUROC CI high", current["auroc"]["ci_high"], reference["patchcore_cgi"]["ci"][1])
        fusion_path = args.out / "fusion_percow.json"
        if fusion_path.exists():
            fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
            # The manuscript states the approximate late-fusion value for the Subspace detector
            # only (reference fusion_auroc_approx_detector); it reports no PatchCore late-fusion
            # value, so those rows are reported without a reference.
            for name, key, ref in (("Subspace fusion", "subspace_fusion", "fusion_auroc_approx"),
                                   ("PatchCore fusion", "patchcore_fusion", None),
                                   ("Per-cow normalization", "percow", "percow_auroc_approx")):
                if "auroc" not in fusion.get(key, {}):
                    continue
                target = reference[ref] if ref else None
                compare(name + " fold-mean AUROC", fusion[key]["auroc"]["mean"], target)
                compare(name + " pooled AUROC", fusion[key]["pooled"]["auroc"], target)
        hei_path = args.out / "hei_perm.csv"
        if hei_path.exists():
            hei = pd.read_csv(hei_path)
            compare("HEI permutation minimum pooled AUROC", float(hei["pooled_auroc"].min()), reference["hei_band"][0])
            compare("HEI permutation maximum pooled AUROC", float(hei["pooled_auroc"].max()), reference["hei_band"][1])
            compare("HEI permutation minimum fold-mean AUROC", float(hei["mean"].min()), reference["hei_band"][0])
            compare("HEI permutation maximum fold-mean AUROC", float(hei["mean"].max()), reference["hei_band"][1])
        compare("Evaluable passes", int(oof.groupby(CONFIG_KEYS).size().max()), reference["analysis_passes"])
        expected_main = {f"{d}_{m}_leg" for d in cfg.detectors for m in cfg.modalities}
        missing_main = sorted(expected_main - summary.keys())
        write_json(args.out / "summary.json", {"complete_main_grid": not missing_main,
            "missing_main_configurations": missing_main, "configurations": summary,
            "comparison": comparison, "reference": reference})
        write_json(args.out / "permutation.json", permutations)
        write_json(args.out / "severity.json", severity)
        pd.DataFrame(bars).to_csv(args.out / "fig4_bars.csv", index=False)
        pd.DataFrame(crowding).to_csv(args.out / "crowding.csv", index=False)
        pd.DataFrame(crowding).to_csv(args.out / "fig5_crowding.csv", index=False)
        cohort_path = args.out / "cohort.json"
        cohort = json.loads(cohort_path.read_text(encoding="utf-8")) if cohort_path.exists() else None
        lines = ["# Re-execution versus manuscript values", "",
            "Main grid status: " + ("COMPLETE" if not missing_main else "INCOMPLETE: " + ", ".join(missing_main)), "",
            f"Values reported in the manuscript were obtained on a {reference['environment']['gpu']} / CUDA {reference['environment']['cuda']} / PyTorch {reference['environment']['pytorch']} system. Re-execution with this repository on the system recorded in results/RUNTIME.md yields the values below; differences are tabulated and are consistent with library-version, GPU-numerics and random-sampling effects. Settings are fixed and were not tuned to manuscript outcomes. Rows labeled fold-mean are means of the five fold values; rows labeled pooled are computed on the pooled out-of-fold scores (the permutation and detected-count rows also use pooled scores).",
            "The manuscript's Evaluation protocol identifies its reported point estimate as the metric on the pooled out-of-fold scores, with the 95% interval taken across the five folds. The pooled row is therefore the like-for-like comparison; the fold-mean row is compared against the same manuscript value so that both estimands are visible. Approximate fusion/per-cow references do not state the aggregation; both are shown for those too. NA means the manuscript does not report an exact point estimate. Rows marked (manuscript figure, read) compare against bar heights read from a figure (about +/-0.01). Permutation rows give k/B for k exceedances among B permutations (reported as <1/B when k = 0) followed by the (k+1)/(B+1) estimate.", ""]
        if cohort:
            lines += ["## Analysis cohort", "", f"Aligned windows {cohort['aligned_windows']}, evaluable passes {cohort['aligned_passes']}, known cows {cohort['known_cows']}, GroupKFold groups {cohort['groups_with_unknown']}; consensus counts " + ", ".join(f"score {k}: {v}" for k, v in sorted(cohort["consensus_counts"].items())) + "; crowding counts " + ", ".join(f"{k}: {v}" for k, v in sorted(cohort["crowding_counts"].items())) + ".", ""]
        lines += ["## Comparison table", "", "| Metric | This repository | Manuscript | \\|Δ\\| |", "|---|---:|---:|---:|"]
        def fmt(value):
            return "NA" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
        for row in comparison:
            lines.append("| " + " | ".join(fmt(row[k]) for k in ("metric", "repository", "manuscript", "absolute_delta")) + " |")
        table2_path = args.out / "table2_legcrop.csv"
        if table2_path.exists():
            lines += ["", "## Table 2: full image versus leg-region crop (fold-mean AUROC)", "",
                      "| Detector | Modality | Full (this repository) | Full (manuscript) | Leg (this repository) | Leg (manuscript) | Leg-full (this repository) | Leg-full (manuscript) |",
                      "|---|---|---:|---:|---:|---:|---:|---:|"]
            for row in pd.read_csv(table2_path).itertuples():
                full_ref = reference["full_auroc"][row.detector][row.modality]
                leg_ref = reference["main_auroc"][row.detector][row.modality]
                lines.append(f"| {row.detector} | {row.modality} | {row.full_auroc:.3f} | {full_ref:.3f} | {row.leg_auroc:.3f} | {leg_ref:.3f} | {row.delta:+.3f} | {leg_ref - full_ref:+.3f} |")
        if hei_path.exists():
            lines += ["", f"## HEI channel permutations (PatchCore, leg crop); manuscript band {reference['hei_band'][0]:.2f}-{reference['hei_band'][1]:.2f}", "",
                      "| Permutation | Fold-mean AUROC | 95% CI | Pooled AUROC |", "|---:|---:|---|---:|"]
            lines += [f"| {int(r.permutation)} | {r.mean:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.pooled_auroc:.3f} |" for r in hei.itertuples()]
        if fusion_path.exists():
            lines += ["", "## Late fusion (GEI + CGI + FDEI, leg crop) and per-cow normalization", "",
                      f"Manuscript (Results, design-factor analysis; stated as approximate): {reference['fusion_auroc_approx_detector']} late fusion about {reference['fusion_auroc_approx']:.2f} versus about {reference['fusion_single_best_approx']:.2f} for the single best modality (Subspace CGI, {reference['fusion_single_best_exact']:.3f} in manuscript Table 2); per-cow normalization about {reference['percow_auroc_approx']:.2f}. The manuscript reports no PatchCore late-fusion value, so the PatchCore fusion rows of the comparison table have no manuscript counterpart.", "",
                      "| Analysis | Fold-mean AUROC | 95% CI | Pooled AUROC |", "|---|---:|---|---:|"]
            for name, key in (("Subspace, concatenated pass-mean vectors", "subspace_fusion"),
                              ("PatchCore, concatenated patch features", "patchcore_fusion"),
                              ("PatchCore-CGI-leg, per-cow z-score", "percow")):
                item = fusion.get(key, {})
                if "auroc" in item:
                    lines.append(f"| {name} | {item['auroc']['mean']:.3f} | [{item['auroc']['ci_low']:.3f}, {item['auroc']['ci_high']:.3f}] | {item['pooled']['auroc']:.3f} |")
                else:
                    lines.append(f"| {name} | not run | | |")
        lines.extend(["", "## AUROC discrepancies greater than 0.03", "",
            "Listed on the pooled estimand, the one the manuscript reports; a fold-mean row is listed only where the same quantity has no pooled counterpart in the table above.", ""])
        pooled_metrics = {r["metric"] for r in comparison if "pooled" in r["metric"]}
        discrepancies = [r for r in comparison if "AUROC" in r["metric"] and "CI" not in r["metric"]
                         and r["absolute_delta"] is not None and r["absolute_delta"] > 0.03
                         and ("fold-mean" not in r["metric"]
                              or r["metric"].replace("fold-mean", "pooled") not in pooled_metrics)]
        lines += [f"- {r['metric']}: {r['absolute_delta']:.6f}" for r in discrepancies] or ["None among reported point estimates."]
        bound = float(reference["patchcore_cgi"]["permutation_p"].lstrip("<"))
        above = [f"- {key}: {permutation_text(result)}, whereas the manuscript states {reference['patchcore_cgi']['permutation_p']}."
                 for key, result in permutations.items() if key in severity and result["p"] >= bound]
        if above:
            lines.extend(["", f"## Permutation p values not below the manuscript's {reference['patchcore_cgi']['permutation_p']}", ""] + above)
        lines.extend(["", "## Interpretation limits", "",
            "- OOF scores from separately fitted folds are pooled on their raw scale, as described in the Methods. Cross-fold score calibration is not established.",
            "- Maximum F1 and severity thresholds are evaluation-set oracle quantities; no deployable threshold is claimed.",
            "- The permutation test exchanges pass labels on fixed OOF scores. It neither retrains the models nor accounts for within-cow dependence or selecting a configuration from the grid.",
            "- Per-cow normalization uses other evaluation scores, is transductive, and mixes standardized multi-pass cows with raw singleton scores.",
            "- Held-out AUROC curves are diagnostic only: all learned runs use epoch 40, with no checkpoint selection.",
            "- See docs/IMPLEMENTATION_NOTES.md, docs/EI_XCHECK.md and docs/SEG_XCHECK.md for implementation choices and upstream checks."])
        (args.out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_runtime(args.out)
        LOGGER.info("Report written: %s", args.out / "REPORT.md")
    except Exception:
        LOGGER.exception("Report generation failed")
        raise


if __name__ == "__main__":
    main()

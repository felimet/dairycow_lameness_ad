"""Regenerate reference energy images from masks and cross-check GEI, CGI, FDEI and HEI_comb6."""
from dataclasses import asdict, replace
import json
import logging
import time
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from _common import arguments, ROOT
from cowlame.config import Config, EnergyConfig
from cowlame.energy_images import synthesize, save_png
from cowlame.evaluate import write_json

LOGGER = logging.getLogger(__name__)
COMPARED = ("GEI", "CGI", "FDEI", "HEI_comb6")
LISTS = ("w30", "gated")
AGGREGATED = ["mae_0_1", "pearson_r", "archived_mean", "regenerated_mean", "archived_max", "regenerated_max"]
METADATA_SOURCE = "mask_meta.json of the pass, copied verbatim"


def cross_check(list_name, metadata, frames, length, cfg, args):
    """Regenerate every window of one keep-list and compare the quantized PNGs with the archived ones."""
    start_time = time.perf_counter()
    keep = [int(window) for window in metadata[f"keep_windows_{list_name}"]]
    destination = args.out / "ei_regen_check" / f"{args.recording:03d}" / f"{args.pass_id:02d}" / list_name
    existing = args.data_root / "matlab_ei" / f"{args.recording:03d}" / f"{args.pass_id:02d}"
    rows = []
    for ordinal, window in enumerate(keep, 1):
        first = (window - cfg.window_origin) * cfg.window_step
        stop = first + length
        if first < 0 or stop > len(frames):
            raise ValueError(f"Window {window} [{first}:{stop}] exceeds mask sequence; provide correct --window-step")
        products = synthesize(frames[first:stop], cfg)
        for modality, values in products.items():
            filename = f"{modality}_{args.recording:03d}_{args.pass_id:02d}_{ordinal:03d}.png"
            save_png(destination / modality / filename, values)
            if modality not in COMPARED:
                continue
            reference_path = existing / modality / filename
            with Image.open(reference_path) as image:
                reference = np.asarray(image, dtype=np.float32) / Config().pixel_max
            if reference.shape != values.shape:
                raise ValueError(f"Reference shape mismatch: {reference_path}")
            # Compare the quantized PNG that is actually delivered.
            actual = np.rint(np.clip(values, 0, 1) * Config().pixel_max) / Config().pixel_max
            rows.append({"modality": modality, "png_ordinal": ordinal, "metadata_window": window,
                "first_frame_zero_based": first, "stop_frame_exclusive": stop,
                "mae_0_1": float(np.abs(actual - reference).mean()),
                "pearson_r": float(pearsonr(actual.ravel(), reference.ravel()).statistic),
                "archived_mean": float(reference.mean()), "regenerated_mean": float(actual.mean()),
                "archived_max": float(reference.max()), "regenerated_max": float(actual.max())})
        LOGGER.info("%s: regenerated kept window %d -> PNG ordinal %d", list_name, window, ordinal)
    table = pd.DataFrame(rows)
    aggregate = table.groupby("modality")[AGGREGATED].mean()
    suffix = "" if list_name == args.window_list else f"_{list_name}"
    table.to_csv(args.out / f"ei_xcheck{suffix}.csv", index=False)
    write_json(args.out / f"ei_xcheck{suffix}.json", {"configuration": asdict(cfg), "window_size": length,
        "window_list": list_name, "n_frames": len(frames), "metadata_source": METADATA_SOURCE, "metadata": metadata,
        "elapsed_seconds": time.perf_counter() - start_time, "mean_per_window": aggregate.to_dict(orient="index")})
    return keep, table, aggregate


def repo_relative(path):
    """Repository-relative form for paths inside the repository, so the docs carry no local absolute path."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def quoted(text):
    return f'"{text}"' if " " in text else text


def summary_table(aggregate):
    lines = ["| Modality | Mean window MAE [0,1] | Mean window Pearson r | Archived mean / max | Regenerated mean / max |",
             "|---|---:|---:|---:|---:|"]
    for modality, row in aggregate.iterrows():
        lines.append(f"| {modality} | {row.mae_0_1:.6f} | {row.pearson_r:.6f} | {row.archived_mean:.3f} / {row.archived_max:.3f} | {row.regenerated_mean:.3f} / {row.regenerated_max:.3f} |")
    return lines


def main():
    try:
        parser = arguments(__doc__)
        parser.add_argument("--recording", type=int, default=1)
        parser.add_argument("--pass-id", type=int, default=1)
        parser.add_argument("--window-step", type=int, default=EnergyConfig().window_step)
        parser.add_argument("--window-size", type=int, default=None)
        parser.add_argument("--window-list", choices=LISTS, default="w30",
                            help="mask_meta.json keep-list to iterate (keep_windows_<name>)")
        parser.add_argument("--compare-list", choices=LISTS, default=None,
                            help="also run this keep-list and append its table to docs/EI_XCHECK.md")
        parser.add_argument("--front-direction", choices=["auto", "left", "right"], default=EnergyConfig().front_direction)
        args = parser.parse_args()
        if args.compare_list == args.window_list:
            raise ValueError("--compare-list must differ from --window-list")
        logging.basicConfig(level=logging.INFO)
        cfg = replace(EnergyConfig(), window_step=args.window_step, front_direction=args.front_direction)
        LOGGER.info("Energy settings: %s; common settings: %s", asdict(cfg), asdict(Config()))
        source = args.data_root / "masks" / f"{args.recording:03d}" / f"{args.pass_id:02d}"
        metadata = json.loads((source / "mask_meta.json").read_text(encoding="utf-8"))
        paths = sorted((source / "frames").glob("frame_*.png"))
        frame_numbers = [int(path.stem.split("_")[-1]) for path in paths]
        if frame_numbers != list(range(len(paths))):
            raise ValueError("Mask files must be contiguous and zero-indexed; explicit remapping required")
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(np.asarray(image.convert("L")) > 0)
        frames = np.stack(frames)
        length = args.window_size or int(metadata["window_size"])
        keep, table, aggregate = cross_check(args.window_list, metadata, frames, length, cfg, args)
        step_frames = (metadata.get("gait") or {}).get("step_frames")
        step_note = (f" Gait step_frames={step_frames} cannot generate those retained indices within {len(frames)} frames."
                     if step_frames and (max(keep) - cfg.window_origin) * step_frames + length > len(frames) else "")
        # The printed command must carry every override this run actually used, including the
        # data root and output folder, or a run against non-default locations cannot be replayed.
        out_text = repo_relative(args.out)
        command = (f"conda run -n cowlame python scripts/regenerate_ei.py "
                   f"--data-root {quoted(repo_relative(args.data_root))} --out {quoted(out_text)} "
                   f"--recording {args.recording} --pass-id {args.pass_id} "
                   f"--window-step {cfg.window_step} --front-direction {cfg.front_direction} --window-list {args.window_list}"
                   + (f" --window-size {length}" if args.window_size else "")
                   + (f" --compare-list {args.compare_list}" if args.compare_list else ""))
        window_size_note = (f"Window size {length} frames, set by the `--window-size` override "
                            f"(mask_meta.json records {int(metadata['window_size'])} frames)."
                            if args.window_size else
                            f"Window size {length} frames, taken from mask_meta.json.")
        front_note = ("front direction is inferred from net centroid motion"
                      if cfg.front_direction == "auto" else
                      f"front direction is fixed to {cfg.front_direction} by the `--front-direction` override")
        worst = aggregate.mae_0_1.idxmax()
        lines = ["# Energy-image cross-check", "", f"Recording {args.recording:03d}, pass {args.pass_id:02d}; {len(frames)} masks; {len(keep)} retained windows (`keep_windows_{args.window_list}`).",
            ""] + summary_table(aggregate)
        lines += ["", f"Command: `{command}`.",
            "", f"{window_size_note} Metadata omit the source window hop. The exposed {cfg.window_step}-frame hop and one-based window origin are implementation assumptions; the mask metadata do not record them.",
            f"Window list `keep_windows_{args.window_list}` of mask_meta.json: its {len(keep)} retained metadata indices {min(keep)}..{max(keep)} map in order to the {len(keep)} PNG ordinals 1..{len(keep)}. This ordinal correspondence is supported by counts (results/window_lists.json, written by scripts/check_window_lists.py), but the original temporal provenance is unverified." + step_note,
            "", "GEI uses nonempty centroid-aligned silhouettes; CGI uses equal-index thirds of that alignment. Alignment rounds centroids to pixels and crops the union of centered foreground bounds before resizing. The unknown MATLAB morphology and body-proportion crop are not emulated.",
            f"FDEI uses segment-peak-retained static energy, adds positive previous-minus-current differences, and sets the first previous frame equal to the first current frame. The Methods give no numerical denoising threshold. HEI averages whole/front/rear masks in one trajectory union; {front_note}, and comb6 is rear/front/whole RGB.",
            "", "What matches and what does not: " + "; ".join(f"{m} r = {aggregate.loc[m, 'pearson_r']:.3f}, MAE = {aggregate.loc[m, 'mae_0_1']:.3f}" for m in aggregate.sort_values("pearson_r", ascending=False).index) + ". "
            f"Spatial structure agrees for every modality (all mean r at least {aggregate.pearson_r.min():.3f}); agreement is best for {aggregate.pearson_r.idxmax()} and weakest in intensity for {worst} (largest MAE), whose archived images peak at {aggregate.loc[worst, 'archived_max']:.2f} while the regenerated ones peak at {aggregate.loc[worst, 'regenerated_max']:.2f}, consistent with a different weighting or normalization of the static term in the pipeline that produced the archived images, which cannot be inspected.",
            "Identity is not expected: the archived energy images were produced by the MATLAB stage with morphological cleaning and a body-proportion crop that are not part of these equations, and the window hop is not recorded in the metadata. These measurements characterize a runnable reference implementation of the equations; the main anomaly analysis uses the archived PNGs unchanged.",
            "", f"Per-window values: {out_text}/ei_xcheck.csv. {out_text}/ei_xcheck.json carries the pass's mask_meta.json as a verbatim copy under `metadata` (its `note` field is in Chinese and describes the mask export stage). Generated PNGs: {out_text}/ei_regen_check/ (not versioned)."]
        if args.compare_list:
            keep2, _, aggregate2 = cross_check(args.compare_list, metadata, frames, length, cfg, args)
            higher_r = int((aggregate2.pearson_r > aggregate.pearson_r).sum())
            lower_mae = int((aggregate2.mae_0_1 < aggregate.mae_0_1).sum())
            lines += ["", f"## Second keep-list: `keep_windows_{args.compare_list}`", "",
                f"Same pass, window length and hop; {len(keep2)} retained metadata indices {min(keep2)}..{max(keep2)} map in order to PNG ordinals 1..{len(keep2)}. Per-window values: {out_text}/ei_xcheck_{args.compare_list}.csv; summary: {out_text}/ei_xcheck_{args.compare_list}.json.",
                ""] + summary_table(aggregate2)
            lines += ["", f"For this pass, `keep_windows_{args.compare_list}` gives the higher mean Pearson r in {higher_r} of {len(COMPARED)} compared modalities and the lower mean MAE in {lower_mae} of {len(COMPARED)} than `keep_windows_{args.window_list}` (mean r over the compared modalities {aggregate2.pearson_r.mean():.3f} versus {aggregate.pearson_r.mean():.3f})."]
        lines += ["", f"## Per-window values (`keep_windows_{args.window_list}`)", "",
            "| PNG ordinal | " + " | ".join(f"{m} MAE | {m} r" for m in COMPARED) + " |",
            "|---:|" + "---:|" * (2 * len(COMPARED))]
        wide = table.pivot(index="png_ordinal", columns="modality", values=["mae_0_1", "pearson_r"])
        for ordinal, row in wide.iterrows():
            lines.append(f"| {ordinal} | " + " | ".join(f"{row[('mae_0_1', m)]:.4f} | {row[('pearson_r', m)]:.4f}" for m in COMPARED) + " |")
        (ROOT / "docs/EI_XCHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        LOGGER.exception("Energy-image regeneration failed")
        raise


if __name__ == "__main__":
    main()

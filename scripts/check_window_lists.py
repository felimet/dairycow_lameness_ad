"""Compare manifest window counts per pass with the keep-lists recorded in mask_meta.json."""
import json
import logging
import pandas as pd
from _common import arguments
from cowlame.data import PASS_KEYS
from cowlame.evaluate import write_json

LOGGER = logging.getLogger(__name__)
LISTS = ("keep_windows_w30", "keep_windows_gated", "keep_windows")
HOP_KEYS = ("window_step", "window_hop", "hop", "step")


def main():
    try:
        args = arguments(__doc__).parse_args()
        logging.basicConfig(level=logging.INFO)
        manifest = pd.read_csv(args.data_root / "window_manifest_maskfirst.csv", dtype=str, keep_default_na=False)
        for column in PASS_KEYS:
            manifest[column] = pd.to_numeric(manifest[column], errors="raise").astype(int)
        manifest_windows = manifest.groupby(PASS_KEYS).size()
        rows, keys = [], set()
        for meta_path in sorted((args.data_root / "masks").glob("*/*/mask_meta.json")):
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            keys |= set(metadata)
            recording, pass_id = meta_path.parent.parts[-2:]
            key = (int(recording), int(pass_id))
            gei = args.data_root / "matlab_ei" / recording / pass_id / "GEI"
            row = {"recording": key[0], "pass": key[1],
                   "manifest_windows": int(manifest_windows.get(key, 0)),
                   "gei_png_files": len(list(gei.glob(f"GEI_{recording}_{pass_id}_*.png"))),
                   "window_size": metadata.get("window_size"),
                   "stride_period_frames": (metadata.get("gait") or {}).get("stride_period_frames")}
            for name in LISTS:
                row[f"len_{name}"] = len(metadata[name]) if name in metadata else None
            rows.append(row)
        if not rows:
            raise FileNotFoundError(f"No mask_meta.json under {args.data_root / 'masks'}")
        summary = {"passes_with_metadata": len(rows),
                   "manifest_count_equals": {name: sum(r["manifest_windows"] == r[f"len_{name}"] for r in rows)
                                             for name in LISTS},
                   "gei_png_files_equal_manifest_windows": sum(r["gei_png_files"] == r["manifest_windows"] for r in rows),
                   "window_size_equals_stride_period": sum(r["window_size"] == r["stride_period_frames"] for r in rows),
                   "hop_keys_present": sorted(keys & set(HOP_KEYS)),
                   "metadata_keys": sorted(keys),
                   "passes": rows}
        write_json(args.out / "window_lists.json", summary)
        LOGGER.info("Window lists: %s", {k: v for k, v in summary.items() if k != "passes"})
    except Exception:
        LOGGER.exception("Window-list check failed")
        raise


if __name__ == "__main__":
    main()

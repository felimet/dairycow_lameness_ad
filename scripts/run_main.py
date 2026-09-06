"""Run the fixed main grid, reporting PatchCore/Subspace before learned models."""
import logging
import time
from _common import arguments, initialize
from cowlame.config import Config
from cowlame.data import load_manifest
from cowlame.evaluate import run_configuration, write_json
import torch

LOGGER = logging.getLogger(__name__)


def main():
    try:
        parser = arguments(__doc__)
        parser.add_argument("--detectors", nargs="+", choices=Config().detectors, default=Config().detectors)
        parser.add_argument("--modalities", nargs="+", choices=Config().modalities, default=Config().modalities)
        parser.add_argument("--crop", choices=["full", "leg"], default="leg")
        args = parser.parse_args()
        cfg = initialize(args)
        start = time.perf_counter()
        frame, passes = load_manifest(args.data_root, cfg)
        # Complete both headline CGI runs before remaining modalities.
        jobs = [(d, m) for d in ("patchcore", "subspace") for m in ("CGI",)
                if d in args.detectors and m in args.modalities]
        jobs += [(d, m) for d in cfg.detectors if d in args.detectors for m in args.modalities
                 if (d, m) not in jobs]
        for detector, modality in jobs:
            run_configuration(frame, passes, detector, modality, args.crop, args.data_root,
                              args.out, args.cache, cfg, force=args.force)
        write_json(args.out / "runtime_main.json", {"elapsed_seconds": time.perf_counter() - start,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "cuda": torch.version.cuda, "jobs": jobs})
    except Exception:
        LOGGER.exception("Main analysis failed")
        raise


if __name__ == "__main__":
    main()

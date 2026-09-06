"""Script arguments and repository-local output defaults."""
import argparse
from dataclasses import replace
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cowlame.config import Config, setup_runtime


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("COWLAME_DATA_ROOT", "F:/cow-data/01_data"))
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache")
    parser.add_argument("--unknown-cow-policy", choices=["single-group", "per-pass"], default="single-group")
    parser.add_argument("--patchcore-blocks", type=int, default=Config().patchcore_blocks)
    parser.add_argument("--force", action="store_true")
    return parser


def initialize(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = replace(Config(), unknown_cow_policy=args.unknown_cow_policy, patchcore_blocks=args.patchcore_blocks)
    setup_runtime(cfg)
    return cfg

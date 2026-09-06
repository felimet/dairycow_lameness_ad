"""Capture installed package versions; print a diff by default, replace the pinned files with --write."""
import argparse
import difflib
from importlib.metadata import version
import logging
import platform
import subprocess
import sys
from _common import ROOT
from cowlame.config import Config
import yaml

LOGGER = logging.getLogger(__name__)
TORCH_INDEX = "--extra-index-url https://download.pytorch.org/whl/cu128"
RAW_RECORD, REQUIREMENTS, ENVIRONMENT = "results/pip_freeze_full.txt", "requirements.txt", "environment.yml"


def freeze(*extra):
    return subprocess.run([sys.executable, "-m", "pip", "freeze", "--all", *extra],
                          capture_output=True, text=True, check=True).stdout


def pin(line):
    """Conda-built packages report build-host file:// locations; pin those by their installed version."""
    if " @ file://" not in line:
        return line
    name = line.split(" @ ", 1)[0]
    return f"{name}=={version(name)}"


def render():
    python_version = platform.python_version()
    pins = [pin(line) for line in freeze("--exclude", "cowlame").splitlines()]
    environment = {"name": "cowlame", "channels": ["conda-forge"], "dependencies": [
        f"python={python_version}", "pip", {"pip": [TORCH_INDEX, *pins]}]}
    return {RAW_RECORD: f"# python {python_version}\n" + freeze(),
            REQUIREMENTS: "\n".join([TORCH_INDEX, *pins]) + "\n",
            ENVIRONMENT: yaml.safe_dump(environment, sort_keys=False)}


def main():
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--write", action="store_true", help="replace the tracked files (default: print a diff)")
        args = parser.parse_args()
        logging.basicConfig(level=logging.INFO)
        LOGGER.info("Configuration: %s", Config())
        for name, content in render().items():
            path = ROOT / name
            if args.write:
                path.write_text(content, encoding="utf-8")
                LOGGER.info("Wrote %s", path)
                continue
            current = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            diff = list(difflib.unified_diff(current, content.splitlines(), f"{name} (tracked)",
                                             f"{name} (installed)", lineterm=""))
            print("\n".join(diff) if diff else f"{name}: no change")
    except Exception:
        LOGGER.exception("Environment capture failed")
        raise


if __name__ == "__main__":
    main()

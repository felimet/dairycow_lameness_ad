#!/usr/bin/env bash
set -euo pipefail
trap 'printf "Reproduction failed at line %s\n" "$LINENO" >&2' ERR
cd -- "$(dirname -- "$0")"
data_root="${COWLAME_DATA_ROOT:-F:/cow-data/01_data}"
out="${1:-results}"
for script in make_cohort.py run_main.py run_ablations.py make_report.py; do
    conda run --no-capture-output -n cowlame python "scripts/$script" --data-root "$data_root" --out "$out"
done

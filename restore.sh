#!/bin/bash
# Unzip the per-pair screen predictions. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")"
find . -name '*_predictions.csv.gz' -exec gunzip -f {} +
echo "unzipped the screen predictions"

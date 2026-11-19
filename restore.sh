#!/bin/bash
# Rejoin files split for GitHub's 100 MB limit, unzip the per-pair screen
# predictions, and check the data against DATA_SHA256SUMS. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")"
find . -name '*.part-aa' -print0 | while IFS= read -r -d '' first; do
  f="${first%.part-aa}"
  cat "$f".part-* > "$f" && rm "$f".part-*
  echo "joined $f"
done
find . -name '*_predictions.csv.gz' -exec gunzip -f {} +
shasum -a 256 -c DATA_SHA256SUMS --quiet && echo "data: all files match DATA_SHA256SUMS"

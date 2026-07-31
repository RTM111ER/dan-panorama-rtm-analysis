#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/local
python analysis/dan_panorama_simulation.py 20000000 | tee results/local/baseline_20m.txt
python analysis/dan_panorama_stronger_tests.py \
  --full-trials 100000000 \
  --density-generated 1000000 \
  --json-output results/local/stronger_100m.json
python analysis/dan_panorama_robustness_tests.py \
  --family-trials 100000000 \
  --json-output results/local/robustness_100m.json

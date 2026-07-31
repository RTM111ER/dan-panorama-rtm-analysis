#!/usr/bin/env bash
set -euo pipefail
mkdir -p results/local
python analysis/dan_panorama_simulation.py 100000 | tee results/local/baseline_quick.txt
python analysis/dan_panorama_stronger_tests.py \
  --full-trials 1000000 \
  --density-generated 100000 \
  --json-output results/local/stronger_quick.json
python analysis/dan_panorama_robustness_tests.py \
  --family-trials 1000000 \
  --json-output results/local/robustness_quick.json

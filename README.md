# Dan Panorama RTM — Reproducible Simulation Repository

This repository contains the data, source code, automated tests, and precomputed simulation results for the **Dan Panorama numerical finding**, recorded between July 26 and July 31, 2026.

The repository is designed to reproduce the calculations and test the observed numerical network against several explicit null models and robustness checks.

## Result Summary

All 16 closures implemented in the code are reproduced successfully.

The main results are:

- Observed case: **16/16 closures**.
- Full-event simulation: in **100,000,000 random events**, the maximum score was **3/16**.
- Dependency-aware analysis: after grouping the closures into six structural families, the observed case scored **6/6**, while the maximum in 100,000,000 random events was **1/6**.
- Exact room–date search: the observed date and room combination was the only **16/16** result among **450,000 combinations**.
- Exact hotel–Amir search: only the pair **435–543** produced the complete network among **810,000 combinations**.
- Semantic role permutation: only the original assignment produced **16/16** among all **288 permutations**.
- Broad free-search test: the observed network contained **49 short numerical relations**. Only **19 of 875,662 valid random sets** reached 49 or more, corresponding to approximately **1 in 46,087**.

Across the implemented tests, the observed network is an **extreme numerical anomaly relative to the tested null models**.

The dependency-aware result is especially important: the anomaly remains extreme even after the 16 individual closures are consolidated into six structural families.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test
make quick
```

To run the heavier simulations:

```bash
make heavy
```

The broad free-search simulation is the most computationally intensive component.

## Repository Structure

```text
analysis/                 Simulation and robustness-test code
data/case_data.json       Case data, constants, and source strings
docs/                     Methodology, interpretation, and source documentation
results/precomputed/      Frozen simulation outputs and SHA-256 checksums
tests/                    Unit tests and exact reconstruction checks
scripts/                  Quick and heavy execution scripts
.github/workflows/        Automated GitHub Actions test workflow
```

## Main Results

| Test | Observed Result | Null-Model Result |
|---|---:|---:|
| Full event — 16 closures | 16/16 | Maximum 3/16 in 100M |
| Dependency-aware — 6 families | 6/6 | Maximum 1/6 in 100M |
| Exact room–date grid | 16/16 | Unique winner among 450,000 |
| Exact hotel–Amir grid | 16/16 | Unique winner among 810,000 |
| Semantic role permutations | 16/16 | Unique winner among 288 |
| Broad free-search test | 49 relations | 19/875,662 reached 49+ |

## Dependency-Aware Simulation

The 16 closures were consolidated into six structural families:

1. Base identity
2. Time–place–person
3. Room recovery
4. Observer–time
5. Node recovery
6. External continuation

The observed case completed all six families.

In 100,000,000 random full events:

| Completed Families | Number of Events |
|---:|---:|
| 0 | 99,999,933 |
| 1 | 67 |
| 2–6 | 0 |

The maximum random score was therefore **1/6**, compared with **6/6** for the observed case.

## Exact Hotel–Amir Grid

All combinations were tested across:

```text
Hotel value: 100–999
Amir value: 100–999
Total combinations: 810,000
```

Only one pair produced both the complete 16-closure network and all six structural families:

```text
Hotel value = 435
Amir value = 543
```

## One-Anchor Perturbation Tests

Each core anchor was replaced individually while the remaining structure was held fixed.

The observed value was the unique 16/16 and 6/6 winner for:

- the initial date among 360 tested dates;
- the room and floor among 1,250 combinations;
- the hotel value among 900 values;
- the Amir value among 900 values;
- Eran’s first-name value among 999 values;
- Amir’s coherent age and birth-year pair among 73 combinations;
- Eran’s age among 73 values.

The short phone number was less selective when tested alone: **84 of 9,000** four-digit values preserved its phone-related closure.

## Broad Free-Search Test

The broad test gives each random set substantial freedom to generate short numerical relations, including:

- one number equal to the sum of two others;
- one number equal to the sum of three others;
- equal-sum pairs;
- digit reversals;
- doubling relations;
- subtraction of digit-reading values;
- subtraction of whole-number and digit-by-digit readings.

The observed network produced:

```text
49 short relations
```

Among 875,662 valid random sets:

```text
19 reached 49 or more
Estimated tail rate: 2.17 × 10⁻⁵
Approximately 1 in 46,087
```

## Precomputed Results

The repository includes frozen outputs from the heavy simulations:

```text
results/precomputed/baseline_20m_results.txt
results/precomputed/stronger_100m_results.json
results/precomputed/stronger_100m_report_he.md
results/precomputed/robustness_100m_results.json
results/precomputed/robustness_100m_report_he.md
results/precomputed/MANIFEST.json
results/precomputed/SHA256SUMS.txt
```

The SHA-256 checksums allow the published outputs to be verified against later modification.

## Runtime Environment

The included precomputed results were generated using:

- Python 3.13.5
- NumPy 2.3.5
- Numba 0.65.1
- Linux x86_64

For the exact test definitions and interpretation of the simulations, see:

```text
docs/METHODOLOGY_HE.md
docs/INTERPRETATION_HE.md
docs/SOURCE_STRINGS_HE.md
```

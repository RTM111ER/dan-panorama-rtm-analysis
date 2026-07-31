#!/usr/bin/env python3
"""Dependency-aware and perturbation robustness tests for Dan Panorama RTM.

This module deliberately reduces the 16 closures to six dependent families,
then tests those families under the full-event null. It also runs exact
one-anchor-at-a-time perturbations and an exact 900x900 hotel/person grid.

It imports the canonical conventions from dan_panorama_stronger_tests.py.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np

import dan_panorama_stronger_tests as core

FAMILY_NAMES = [
    "base_identity",
    "time_place_person",
    "room_recovery",
    "observer_time",
    "node_recovery",
    "external_continuation",
]
FAMILY_GROUPS = [
    (0, 1),
    (2, 3, 4, 5),
    (6, 9),
    (7, 8),
    (10, 11),
    (12, 13, 14, 15),
]


def family_flags(conditions: list[bool] | np.ndarray) -> list[bool]:
    return [all(bool(conditions[i]) for i in group) for group in FAMILY_GROUPS]


def case_conditions(
    *,
    H: int = 435,
    A: int = 543,
    E: int = 612,
    first_e: int = 320,
    i1: int | None = None,
    ridx: int | None = None,
    age_a: int = 76,
    birth: int = 1949,
    age_e: int = 44,
    short: int = 5450,
    long_digits: str = "97235202552",
    anchor58: int = 58,
) -> list[bool]:
    if i1 is None:
        i1 = (dt.date(2026, 7, 26) - core.DATES[0]).days
    if ridx is None:
        ridx = int(np.where((core.FLOORS == 12) & (core.SUFFIXES == 6))[0][0])
    i2, i3 = i1 + 4, i1 + 5
    if not (0 <= i1 < 360):
        return [False] * 16

    room = int(core.ROOMS[ridx])
    floor = int(core.FLOORS[ridx])
    B = core.g("חדר מספר") + int(core.ROOM_WORD[ridx]) + H + core.g("תל אביב קומה") + int(core.FLOOR_WORD[ridx])
    Q1 = core.g("מהו מספר חדר ששהה") + A + core.g("במלון") + H
    Q2 = Q1 + core.g("תל אביב")
    age_phrase = E + core.g("בן") + core.g(core.number_words(age_e))
    node = Q2 + A
    route = int(core.DATE_PADDED[i1]) - int(core.ROOM_DESC[ridx]) - A
    short_diff = int(core.WHOLE[short] - core.DIGITS[short])
    long_value = sum(core.g(core.DIGIT_WORD[int(ch)]) for ch in long_digits)

    return [
        B == int(core.DATE_FULL[i1]) + room + floor + int(core.DATE_COMPACT[i1]),
        0 <= B <= core.MAX_N and B - int(core.WHOLE[B]) - int(core.DIGITS[B]) == 2 * int(core.DATE_COMPACT[i1]),
        0 <= route <= core.MAX_N and route == 2 * int(core.DATE_COMPACT[i1]) and int(core.REVERSE[route]) == H,
        int(core.ROOM_FLOOR_PHRASE[ridx]) - 2 * A - int(core.DATE_COMPACT[i1]) - room - floor == H,
        core.YEAR - (B - int(core.DATE_PADDED[i1])) + age_a == A,
        B - birth - core.YEAR + age_a == int(core.DATE_COMPACT[i1]),
        B - Q1 - A - anchor58 == room,
        B - int(core.DATE_NUM[i1]) - int(core.DATE_NUM[i2]) == age_e,
        int(core.DATE_FULL[i1]) + int(core.DATE_FULL[i2]) - B - E == age_e,
        Q2 - age_phrase - floor - first_e == room,
        node == int(core.DATE_FULL[i2]) + room,
        0 <= node <= core.MAX_N and node - int(core.DIGITS[node]) == int(core.ROOM_PHRASE[ridx]),
        room - floor - short_diff == int(core.DATE_COMPACT[i1]),
        0 <= long_value <= core.MAX_N and int(core.REVERSE[long_value]) == int(core.ROOM_FLOOR_PHRASE[ridx]),
        int(core.REVERSE[int(core.DATE_FULL[i3])]) == int(core.SPLIT[ridx]),
        E == int(core.SPLIT[ridx]),
    ]


def full_event_family_monte_carlo(trials: int, seed: int = 2026073103, batch: int = 250_000):
    """Full-event null, scored as six dependent families rather than 16 closures."""
    rng = np.random.default_rng(seed)
    counts = np.zeros(7, dtype=np.int64)
    pattern_counts: collections.Counter[str] = collections.Counter()
    max_score = 0

    for start in range(0, trials, batch):
        n = min(batch, trials - start)
        i1 = rng.integers(0, 360, size=n)
        i2, i3 = i1 + 4, i1 + 5
        ridx = rng.integers(0, len(core.ROOMS), size=n)
        H = rng.integers(100, 1000, size=n)
        surname = rng.integers(50, 401, size=n)
        first_a = rng.integers(20, 501, size=n)
        first_e = rng.integers(20, 501, size=n)
        A, E = first_a + surname, first_e + surname
        age_a = rng.integers(18, 91, size=n)
        birth = core.YEAR - age_a - 1
        age_e = rng.integers(18, 91, size=n)
        short = rng.integers(1000, 10000, size=n)
        suffix = rng.integers(0, 10, size=(n, 7))

        floor, room = core.FLOORS[ridx], core.ROOMS[ridx]
        B = core.g("חדר מספר") + core.ROOM_WORD[ridx] + H + core.g("תל אביב קומה") + core.FLOOR_WORD[ridx]
        Q1 = core.g("מהו מספר חדר ששהה") + A + core.g("במלון") + H
        Q2 = Q1 + core.g("תל אביב")
        node = Q2 + A
        age_phrase = E + core.g("בן") + core.AGE_WORD[age_e]
        route = core.DATE_PADDED[i1] - core.ROOM_DESC[ridx] - A
        short_diff = core.WHOLE[short] - core.DIGITS[short]
        long_value = core.PREFIX_VALUE + core.DIGIT_VALUES[suffix].sum(axis=1)

        c = np.empty((16, n), dtype=np.bool_)
        c[0] = B == core.DATE_FULL[i1] + room + floor + core.DATE_COMPACT[i1]
        c[1] = False
        valid = (B >= 0) & (B <= core.MAX_N)
        c[1, valid] = B[valid] - core.WHOLE[B[valid]] - core.DIGITS[B[valid]] == 2 * core.DATE_COMPACT[i1][valid]
        c[2] = False
        valid = (route >= 0) & (route <= core.MAX_N)
        c[2, valid] = (route[valid] == 2 * core.DATE_COMPACT[i1][valid]) & (core.REVERSE[route[valid]] == H[valid])
        c[3] = core.ROOM_FLOOR_PHRASE[ridx] - 2 * A - core.DATE_COMPACT[i1] - room - floor == H
        c[4] = core.YEAR - (B - core.DATE_PADDED[i1]) + age_a == A
        c[5] = B - birth - core.YEAR + age_a == core.DATE_COMPACT[i1]
        c[6] = B - Q1 - A - 58 == room
        c[7] = B - core.DATE_NUM[i1] - core.DATE_NUM[i2] == age_e
        c[8] = core.DATE_FULL[i1] + core.DATE_FULL[i2] - B - E == age_e
        c[9] = Q2 - age_phrase - floor - first_e == room
        c[10] = node == core.DATE_FULL[i2] + room
        c[11] = False
        valid = (node >= 0) & (node <= core.MAX_N)
        c[11, valid] = node[valid] - core.DIGITS[node[valid]] == core.ROOM_PHRASE[ridx][valid]
        c[12] = room - floor - short_diff == core.DATE_COMPACT[i1]
        c[13] = False
        valid = (long_value >= 0) & (long_value <= core.MAX_N)
        c[13, valid] = core.REVERSE[long_value[valid]] == core.ROOM_FLOOR_PHRASE[ridx][valid]
        c[14] = core.REVERSE[core.DATE_FULL[i3]] == core.SPLIT[ridx]
        c[15] = E == core.SPLIT[ridx]

        families = np.vstack([np.all(c[list(group)], axis=0) for group in FAMILY_GROUPS])
        score = families.sum(axis=0)
        counts += np.bincount(score, minlength=7)
        max_score = max(max_score, int(score.max()))
        # Record non-zero family patterns only; useful for diagnosing dependence.
        nonzero = np.where(score > 0)[0]
        for idx in nonzero:
            key = "".join("1" if families[j, idx] else "0" for j in range(6))
            pattern_counts[key] += 1

    return counts, max_score, pattern_counts


def exact_hotel_amir_grid():
    """Exhaust all 810,000 H/A pairs, leaving every other observed anchor fixed."""
    hist_raw: collections.Counter[int] = collections.Counter()
    hist_family: collections.Counter[int] = collections.Counter()
    winners_raw: list[dict] = []
    winners_family: list[dict] = []
    for H in range(100, 1000):
        for A in range(100, 1000):
            c = case_conditions(H=H, A=A)
            raw, fam = sum(c), sum(family_flags(c))
            hist_raw[raw] += 1
            hist_family[fam] += 1
            if raw == 16:
                winners_raw.append({"hotel_value": H, "amir_value": A})
            if fam == 6:
                winners_family.append({"hotel_value": H, "amir_value": A})
    return hist_raw, hist_family, winners_raw, winners_family


def one_anchor_perturbations():
    """Exact local/global perturbation of one external anchor at a time."""
    observed_c = case_conditions()
    observed_raw = sum(observed_c)
    observed_family = sum(family_flags(observed_c))
    ridx_obs = int(np.where((core.FLOORS == 12) & (core.SUFFIXES == 6))[0][0])
    i1_obs = (dt.date(2026, 7, 26) - core.DATES[0]).days

    spaces: dict[str, list[tuple[object, list[bool]]]] = {}
    spaces["initial_date"] = [(core.DATES[i].isoformat(), case_conditions(i1=i)) for i in range(360)]
    spaces["room_floor"] = [
        ({"floor": int(core.FLOORS[r]), "suffix": int(core.SUFFIXES[r]), "room": int(core.ROOMS[r])}, case_conditions(ridx=r))
        for r in range(len(core.ROOMS))
    ]
    spaces["hotel_value"] = [(h, case_conditions(H=h)) for h in range(100, 1000)]
    spaces["amir_value"] = [(a, case_conditions(A=a)) for a in range(100, 1000)]
    spaces["eran_first_value"] = [(f, case_conditions(first_e=f, E=f + 292)) for f in range(1, 1000)]
    spaces["amir_age_birth_coherent"] = [
        ({"age": a, "birth": core.YEAR - a - 1}, case_conditions(age_a=a, birth=core.YEAR - a - 1))
        for a in range(18, 91)
    ]
    spaces["eran_age"] = [(a, case_conditions(age_e=a)) for a in range(18, 91)]
    spaces["short_phone"] = [(p, case_conditions(short=p)) for p in range(1000, 10000)]

    results = {}
    for name, values in spaces.items():
        raw_hist: collections.Counter[int] = collections.Counter()
        fam_hist: collections.Counter[int] = collections.Counter()
        raw_best = -1
        fam_best = -1
        raw_winners = []
        fam_winners = []
        for label, c in values:
            raw, fam = sum(c), sum(family_flags(c))
            raw_hist[raw] += 1
            fam_hist[fam] += 1
            if raw > raw_best:
                raw_best, raw_winners = raw, [label]
            elif raw == raw_best:
                raw_winners.append(label)
            if fam > fam_best:
                fam_best, fam_winners = fam, [label]
            elif fam == fam_best:
                fam_winners.append(label)
        results[name] = {
            "space_size": len(values),
            "raw_histogram": dict(sorted(raw_hist.items())),
            "family_histogram": dict(sorted(fam_hist.items())),
            "best_raw_score": raw_best,
            "best_raw_values": raw_winners[:25],
            "best_raw_tie_count": len(raw_winners),
            "best_family_score": fam_best,
            "best_family_values": fam_winners[:25],
            "best_family_tie_count": len(fam_winners),
        }

    results["observed"] = {
        "raw_score": observed_raw,
        "family_score": observed_family,
        "date": core.DATES[i1_obs].isoformat(),
        "room": int(core.ROOMS[ridx_obs]),
    }
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family-trials", type=int, default=1_000_000)
    p.add_argument("--json-output", default="")
    args = p.parse_args()
    started = time.time()

    obs = core.observed_conditions()
    obs_families = family_flags(obs)
    family_hist, family_max, patterns = full_event_family_monte_carlo(args.family_trials)
    ha_raw, ha_fam, ha_raw_winners, ha_fam_winners = exact_hotel_amir_grid()
    perturb = one_anchor_perturbations()

    out = {
        "observed": {
            "closure_score": sum(obs),
            "closure_flags": obs,
            "family_score": sum(obs_families),
            "family_flags": dict(zip(FAMILY_NAMES, obs_families)),
        },
        "dependency_aware_full_event": {
            "trials": args.family_trials,
            "histogram": {str(i): int(c) for i, c in enumerate(family_hist) if c},
            "maximum_family_score": family_max,
            "zero_hit_95pct_upper_bound_for_6_of_6": 3 / args.family_trials,
            "nonzero_family_patterns": dict(patterns.most_common()),
        },
        "exact_hotel_amir_grid": {
            "combinations": 900 * 900,
            "raw_histogram": dict(sorted(ha_raw.items())),
            "family_histogram": dict(sorted(ha_fam.items())),
            "raw_16_of_16_winners": ha_raw_winners,
            "family_6_of_6_winners": ha_fam_winners,
        },
        "one_anchor_perturbations": perturb,
        "runtime_seconds": time.time() - started,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        Path(args.json_output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

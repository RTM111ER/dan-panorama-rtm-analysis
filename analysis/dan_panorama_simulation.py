#!/usr/bin/env python3
"""
Dan Panorama RTM: reproducible null-model checks.

The script uses the exact formulas and Hebrew gematria conventions appearing
in the supplied transcript. It performs:
1. Exact enumeration of several number/date/phone transformations.
2. A Monte Carlo test in which semantic anchors are randomized while the
   formula templates remain frozen.

Usage:
    python dan_panorama_simulation.py [number_of_trials]
Default: 1,000,000 trials.
"""

from __future__ import annotations

import calendar
import datetime as dt
import sys
from collections import Counter
import numpy as np

GEM = {
    "א":1,"ב":2,"ג":3,"ד":4,"ה":5,"ו":6,"ז":7,"ח":8,"ט":9,
    "י":10,"כ":20,"ך":20,"ל":30,"מ":40,"ם":40,"נ":50,"ן":50,"ס":60,
    "ע":70,"פ":80,"ף":80,"צ":90,"ץ":90,"ק":100,"ר":200,"ש":300,"ת":400,
}

def g(text: str) -> int:
    return sum(GEM.get(ch, 0) for ch in text)

UNITS = {0:"",1:"אחד",2:"שתיים",3:"שלוש",4:"ארבע",5:"חמש",6:"שש",7:"שבע",8:"שמונה",9:"תשע"}
TEENS = {
    10:"עשר",11:"אחת עשרה",12:"שתים עשרה",13:"שלוש עשרה",14:"ארבע עשרה",
    15:"חמש עשרה",16:"שש עשרה",17:"שבע עשרה",18:"שמונה עשרה",19:"תשע עשרה",
}
TENS = {20:"עשרים",30:"שלושים",40:"ארבעים",50:"חמישים",60:"שישים",70:"שבעים",80:"שמונים",90:"תשעים"}
HUNDREDS = {100:"מאה",200:"מאתיים",300:"שלוש מאות",400:"ארבע מאות",500:"חמש מאות",
            600:"שש מאות",700:"שבע מאות",800:"שמונה מאות",900:"תשע מאות"}
THOUSANDS = {1000:"אלף",2000:"אלפיים",3000:"שלושת אלפים",4000:"ארבעת אלפים",
             5000:"חמשת אלפים",6000:"ששת אלפים",7000:"שבעת אלפים",
             8000:"שמונת אלפים",9000:"תשעת אלפים"}
DIGIT_WORD = {0:"אפס",1:"אחד",2:"שתיים",3:"שלוש",4:"ארבע",5:"חמש",6:"שש",7:"שבע",8:"שמונה",9:"תשע"}
DIGIT_VALUE = np.array([g(DIGIT_WORD[d]) for d in range(10)], dtype=int)

def under_100(n: int) -> str:
    if n == 0:
        return ""
    if n < 10:
        return UNITS[n]
    if n < 20:
        return TEENS[n]
    tens = (n // 10) * 10
    unit = n % 10
    return TENS[tens] + ((" ו" + UNITS[unit]) if unit else "")

def number_words(n: int) -> str:
    if not 0 <= n <= 9999:
        raise ValueError("number_words supports 0..9999")
    if n == 0:
        return "אפס"
    parts: list[str] = []
    thousands = (n // 1000) * 1000
    remainder = n % 1000
    if thousands:
        parts.append(THOUSANDS[thousands])
    hundreds = (remainder // 100) * 100
    remainder %= 100
    if hundreds:
        parts.append(HUNDREDS[hundreds])
    if remainder:
        words = under_100(remainder)
        if remainder < 10 and (thousands or hundreds):
            words = "ו" + words
        parts.append(words)
    return " ".join(parts)

def digit_g(n: int) -> int:
    return sum(g(DIGIT_WORD[int(ch)]) for ch in str(n))

# Date convention used in the transcript.
MONTHS = {
    1:"בינואר",2:"בפברואר",3:"במרץ",4:"באפריל",5:"במאי",6:"ביוני",
    7:"ביולי",8:"באוגוסט",9:"בספטמבר",10:"באוקטובר",11:"בנובמבר",12:"בדצמבר",
}
YEAR_2026 = "אלפיים עשרים ושש"

def date_value_2026(month: int, day: int) -> int:
    return g(f"{under_100(day)} {MONTHS[month]} {YEAR_2026}")

def reverse_int(n: int) -> int:
    return int(str(n)[::-1])

def exact_checks() -> dict:
    # Values in transcript:
    # B = 3731 + H, Q1 = 1381 + A + H, Q2 = 1826 + A + H.
    hotel_hits = []
    for H in range(100, 1000):
        B = 3731 + H
        conditions = [
            B == 2681 + 1206 + 12 + 267,
            B - g(number_words(B)) - digit_g(B) == 534,
            B - 1949 - 2026 + 76 == 267,
            B - (26 + 7 + 2026) - (30 + 7 + 2026) == 44,
        ]
        if all(conditions):
            hotel_hits.append(H)

    amir_hits = []
    H = 435
    B = 3731 + H
    for A in range(100, 1000):
        Q1 = 1381 + A + H
        Q2 = 1826 + A + H
        r = 2607 - 1530 - A
        conditions = [
            r == 534 and reverse_int(r) == H,
            3006 - 2*A - 267 - 1206 - 12 == H,
            2026 - (B - 2607) + 76 == A,
            B - Q1 - A - 58 == 1206,
            Q2 + A == 2141 + 1206,
        ]
        if all(conditions):
            amir_hits.append(A)

    eran_first_hits = []
    surname = 292
    A = 543
    Q2 = 1826 + A + H
    for first in range(1, 1000):
        full = first + surname
        age_phrase = full + 654
        conditions = [
            Q2 - age_phrase - 12 - first == 1206,
            2681 + 2141 - B - full == 44,
            reverse_int(2160) == full,
            full == 612,
        ]
        if all(conditions):
            eran_first_hits.append(first)

    base_transform_hits = [
        n for n in range(1000, 7000)
        if n - g(number_words(n)) - digit_g(n) == 534
    ]
    node_transform_hits = [
        n for n in range(1000, 7000)
        if n - digit_g(n) == 1430
    ]
    short_phone_hits = [
        n for n in range(1000, 10000)
        if g(number_words(n)) - digit_g(n) == 927
    ]

    # Exact DP for an 11-digit number with fixed 9723 prefix and seven free digits.
    prefix_value = sum(DIGIT_VALUE[int(ch)] for ch in "9723")
    distribution = Counter({prefix_value: 1})
    for _ in range(7):
        new = Counter()
        for total, count in distribution.items():
            for d in range(10):
                new[total + int(DIGIT_VALUE[d])] += count
        distribution = new
    long_phone_count = distribution[6003]
    long_phone_total = 10_000_000

    date_hits = []
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(2026, month)[1] + 1):
            value = date_value_2026(month, day)
            if reverse_int(value) == 612:
                date_hits.append((dt.date(2026, month, day).isoformat(), value))

    return {
        "hotel_bundle_hits_100_to_999": hotel_hits,
        "amir_bundle_hits_100_to_999": amir_hits,
        "eran_first_bundle_hits_1_to_999": eran_first_hits,
        "base_transform_hits_1000_to_6999": base_transform_hits,
        "node_transform_hits_1000_to_6999": node_transform_hits,
        "short_phone_hits": len(short_phone_hits),
        "short_phone_total": 9000,
        "short_phone_probability": len(short_phone_hits) / 9000,
        "long_phone_hits_prefix_9723": long_phone_count,
        "long_phone_total_prefix_9723": long_phone_total,
        "long_phone_probability_prefix_9723": long_phone_count / long_phone_total,
        "date_reverse_612_hits_in_2026": date_hits,
    }

def monte_carlo(trials: int, seed: int = 20260731) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)

    H_values = np.arange(100, 1000)
    B_values = 3731 + H_values
    B_whole = np.array([g(number_words(int(n))) for n in B_values])
    B_digits = np.array([digit_g(int(n)) for n in B_values])

    short_lookup = np.zeros(10000, dtype=bool)
    for n in range(1000, 10000):
        short_lookup[n] = g(number_words(n)) - digit_g(n) == 927

    node_lookup = np.zeros(10000, dtype=bool)
    for n in range(1000, 10000):
        node_lookup[n] = n - digit_g(n) == 1430

    date_reversals = []
    for month in range(1, 13):
        for day in range(1, calendar.monthrange(2026, month)[1] + 1):
            date_reversals.append(reverse_int(date_value_2026(month, day)))
    date_reversals = np.array(date_reversals, dtype=int)

    score_counts = np.zeros(16, dtype=np.int64)
    batch_size = 250_000

    for start in range(0, trials, batch_size):
        size = min(batch_size, trials - start)
        H = rng.integers(100, 1000, size=size)
        A = rng.integers(100, 1000, size=size)
        first = rng.integers(1, 1000, size=size)
        date_index = rng.integers(0, 365, size=size)
        short = rng.integers(1000, 10000, size=size)
        suffix = rng.integers(0, 10, size=(size, 7))

        B = 3731 + H
        Q1 = 1381 + A + H
        Q2 = 1826 + A + H
        full = first + 292
        age_phrase = full + 654
        node = Q2 + A
        result = 2607 - 1530 - A

        reversed_result = (
            (result % 10) * 100
            + ((result // 10) % 10) * 10
            + (result // 100)
        )
        long_sum = (
            sum(int(DIGIT_VALUE[int(ch)]) for ch in "9723")
            + DIGIT_VALUE[suffix].sum(axis=1)
        )

        score = np.zeros(size, dtype=np.int8)
        score += B == 4166
        score += B - B_whole[H - 100] - B_digits[H - 100] == 534
        score += B - 1949 - 2026 + 76 == 267
        score += B - (26 + 7 + 2026) - (30 + 7 + 2026) == 44
        score += (result == 534) & (reversed_result == H)
        score += 3006 - 2*A - 267 - 1206 - 12 == H
        score += 2026 - (B - 2607) + 76 == A
        score += B - Q1 - A - 58 == 1206
        score += Q2 + A == 2141 + 1206
        score += Q2 - age_phrase - 12 - first == 1206
        score += 2681 + 2141 - B - full == 44
        valid_node = (node >= 1000) & (node < 10000)
        node_condition = np.zeros(size, dtype=bool)
        node_condition[valid_node] = node_lookup[node[valid_node]]
        score += node_condition
        score += short_lookup[short]
        score += long_sum == 6003
        score += (date_reversals[date_index] == 612) & (full == 612)

        score_counts += np.bincount(score, minlength=16)

    return score_counts, int(np.max(np.nonzero(score_counts)[0]))

def main() -> None:
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    checks = exact_checks()
    print("EXACT ENUMERATIONS")
    for key, value in checks.items():
        print(f"{key}: {value}")

    observed_score = 15
    print(f"\nObserved protocol score: {observed_score}/15")
    print(f"Running Monte Carlo: {trials:,} trials")
    score_counts, maximum = monte_carlo(trials)
    print(f"Maximum simulated score: {maximum}/15")
    print("Score distribution:")
    for score, count in enumerate(score_counts):
        if count:
            print(f"  {score}: {count:,}")
    print(f"Trials reaching 15/15: {score_counts[15]:,}")

if __name__ == "__main__":
    main()

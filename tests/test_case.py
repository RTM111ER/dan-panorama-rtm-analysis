from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

import dan_panorama_stronger_tests as strong
import dan_panorama_robustness_tests as robust


class GematriaTests(unittest.TestCase):
    def test_core_values(self):
        expected = {
            "דן פנורמה": 435,
            "אמיר הרפז": 543,
            "ערן": 320,
            "ערן הרפז": 612,
            "חדר מספר אלף מאתיים ושש דן פנורמה תל אביב קומה שתים עשרה": 4166,
            "עשרים ושש ביולי אלפיים עשרים ושש": 2681,
            "שלושים ביולי אלפיים עשרים ושש": 2141,
            "שלושים ואחד ביולי אלפיים עשרים ושש": 2160,
            "מהו מספר חדר ששהה אמיר הרפז במלון דן פנורמה": 2359,
            "מהו מספר חדר ששהה אמיר הרפז במלון דן פנורמה תל אביב": 2804,
        }
        for text, value in expected.items():
            with self.subTest(text=text):
                self.assertEqual(strong.g(text), value)

    def test_all_observed_closures(self):
        flags = strong.observed_conditions()
        self.assertEqual(len(flags), 16)
        self.assertTrue(all(flags), flags)

    def test_dependency_aware_families(self):
        flags = robust.family_flags(strong.observed_conditions())
        self.assertEqual(flags, [True] * 6)

    def test_exact_transforms(self):
        checks = __import__("dan_panorama_simulation").exact_checks()
        self.assertEqual(checks["hotel_bundle_hits_100_to_999"], [435])
        self.assertEqual(checks["amir_bundle_hits_100_to_999"], [543])
        self.assertEqual(checks["eran_first_bundle_hits_1_to_999"], [320])
        self.assertEqual(checks["date_reverse_612_hits_in_2026"], [("2026-07-31", 2160)])


if __name__ == "__main__":
    unittest.main()

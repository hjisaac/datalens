from __future__ import annotations

import math
import unittest

from src.sequence_token_stats import analyze_strings, compute_token_statistics


class TestComputeTokenStatistics(unittest.TestCase):
    def test_empty_corpus(self) -> None:
        report = compute_token_statistics([])
        self.assertEqual(report["meta"]["total_sequences"], 0)
        self.assertEqual(report["meta"]["total_tokens"], 0)
        self.assertEqual(report["meta"]["num_token_types"], 0)
        self.assertEqual(report["corpus_token_mass"]["per_token"], [])
        self.assertEqual(report["sequence_presence"]["per_token"], [])

    def test_skips_empty_rows(self) -> None:
        report = compute_token_statistics([[], ["a"], []])
        self.assertEqual(report["meta"]["total_sequences"], 1)
        self.assertEqual(report["meta"]["total_tokens"], 1)

    def test_simple_two_sequences_counts(self) -> None:
        rows = [["a", "b"], ["a", "a", "b"]]
        report = compute_token_statistics(rows)
        self.assertEqual(report["meta"]["total_sequences"], 2)
        self.assertEqual(report["meta"]["total_tokens"], 5)
        self.assertEqual(report["meta"]["num_token_types"], 2)

        by_token = {r["token"]: r for r in report["corpus_token_mass"]["per_token"]}
        self.assertEqual(by_token["a"]["count"], 3)
        self.assertEqual(by_token["b"]["count"], 2)
        self.assertAlmostEqual(by_token["a"]["fraction_of_tokens"], 0.6)
        self.assertAlmostEqual(by_token["b"]["fraction_of_tokens"], 0.4)

    def test_single_token_type_entropy_zero(self) -> None:
        report = compute_token_statistics([["x"], ["x", "x"]])
        self.assertEqual(report["corpus_token_mass"]["corpus_entropy"], 0.0)
        self.assertEqual(report["corpus_token_mass"]["max_token_fraction"], 1.0)

    def test_report_top_level_keys(self) -> None:
        report = compute_token_statistics([["a"]])
        self.assertEqual(
            set(report.keys()),
            {"meta", "corpus_token_mass", "sequence_presence", "intra_sequence"},
        )
        for block in (
            report["corpus_token_mass"],
            report["sequence_presence"],
            report["intra_sequence"],
        ):
            self.assertIn("bounds", block)

    def test_sequence_presence_fractions(self) -> None:
        rows = [["a", "b"], ["b", "c"]]
        report = compute_token_statistics(rows)
        by_token = {r["token"]: r for r in report["sequence_presence"]["per_token"]}
        self.assertAlmostEqual(by_token["a"]["fraction_of_sequences"], 0.5)
        self.assertAlmostEqual(by_token["b"]["fraction_of_sequences"], 1.0)
        self.assertAlmostEqual(by_token["c"]["fraction_of_sequences"], 0.5)

    def test_log_base_meta(self) -> None:
        report = compute_token_statistics([["a", "b"]], log_base=2.0)
        self.assertEqual(report["meta"]["log_base"], "2.0")

    def test_quantile_sample_cap_passes_through(self) -> None:
        report = compute_token_statistics([["a", "b"]], quantile_sample_cap=123)
        h = report["intra_sequence"]["sequence_entropy_H"]
        self.assertEqual(h["reservoir_size_cap"], 123)


class TestAnalyzeStrings(unittest.TestCase):
    def test_matches_manual_tokenization(self) -> None:
        strings = ["a b", "a a b"]

        def tokenize(s: str) -> list[str]:
            return s.split()

        r1 = analyze_strings(strings, tokenize)
        r2 = compute_token_statistics([["a", "b"], ["a", "a", "b"]])
        self.assertEqual(r1["meta"], r2["meta"])
        self.assertEqual(r1["corpus_token_mass"], r2["corpus_token_mass"])

    def test_skips_none_strings(self) -> None:
        def tokenize(s: str) -> list[str]:
            return s.split()

        report = analyze_strings(["a b", None, "a"], tokenize)  # type: ignore[list-item]
        self.assertEqual(report["meta"]["total_sequences"], 2)

    def test_json_serializable_meta_numbers(self) -> None:
        report = analyze_strings(["x y", "x"], lambda s: s.split())
        meta = report["meta"]
        self.assertIsInstance(meta["total_sequences"], int)
        self.assertIsInstance(meta["total_tokens"], int)
        self.assertIsInstance(meta["num_token_types"], int)


class TestIntraSequence(unittest.TestCase):
    def test_normalized_entropy_in_zero_one_when_distinct_gt_one(self) -> None:
        rows = [["a", "b", "c"], ["a", "b", "c"]]
        report = compute_token_statistics(rows)
        j = report["intra_sequence"]["normalized_entropy_J"]
        self.assertIsNotNone(j["max"])
        if j["max"] is not None:
            self.assertGreaterEqual(j["max"], 0.0)
            self.assertLessEqual(j["max"], 1.0 + 1e-9)

    def test_intra_has_sequence_entropy_summary(self) -> None:
        report = compute_token_statistics([["a", "b"]], log_base=math.e)
        h = report["intra_sequence"]["sequence_entropy_H"]
        self.assertIn("mean", h)
        self.assertIn("quantiles_reservoir", h)


if __name__ == "__main__":
    unittest.main()

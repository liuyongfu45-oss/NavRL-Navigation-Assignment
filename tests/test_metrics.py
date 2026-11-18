"""Tests for metrics aggregation, CSV export and figures."""
import csv
import os
import tempfile
import unittest

from navrl.metrics import (EpisodeRecord, aggregate, plot_summary,
                            write_csv)


def _make_record(scenario: str, reached: bool, collided: bool,
                 reward: float, path: float, time_s: float) -> EpisodeRecord:
    return EpisodeRecord(
        scenario=scenario, seed=0, robot_id=0, reached=reached,
        collided=collided, timeout=not reached and not collided,
        steps=int(time_s / 0.1), time_s=time_s, path_length=path,
        total_reward=reward, min_clearance=1.2, control_effort=0.5,
        jerk=0.3, reward_curve=[reward],
    )


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.records = [
            _make_record("S1_dynamic", True, False, 8.6, 36.0, 18.7),
            _make_record("S1_dynamic", True, False, 8.4, 35.2, 18.1),
            _make_record("S1_dynamic", False, True, -6.0, 12.0, 5.0),
            _make_record("S2_static", True, False, 8.9, 37.0, 19.3),
            _make_record("S2_static", True, False, 8.0, 36.4, 19.0),
        ]

    def test_aggregate_rates(self):
        summary = aggregate(self.records)
        self.assertAlmostEqual(summary["S1_dynamic"]["success_rate"], 2 / 3)
        self.assertAlmostEqual(summary["S1_dynamic"]["collision_rate"], 1 / 3)
        self.assertEqual(summary["S2_static"]["episodes"], 2)
        self.assertGreater(summary["S2_static"]["mean_reward"], 0.0)

    def test_csv_roundtrip(self):
        summary = aggregate(self.records)
        with tempfile.TemporaryDirectory() as tmp:
            detail, summary_path = write_csv(self.records, summary, tmp)
            self.assertTrue(os.path.exists(detail))
            self.assertTrue(os.path.exists(summary_path))
            with open(detail, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), len(self.records))
            with open(summary_path, encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 2)

    def test_plot_summary_creates_figures(self):
        summary = aggregate(self.records)
        with tempfile.TemporaryDirectory() as tmp:
            figures = plot_summary(summary, self.records, tmp)
            self.assertEqual(len(figures), 4)
            for path in figures:
                self.assertTrue(os.path.exists(path))
                self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()

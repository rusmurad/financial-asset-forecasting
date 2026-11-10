import ast
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


class QualityAuditTests(unittest.TestCase):
    def setUp(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "eda.ipynb"
        notebook = json.loads(notebook_path.read_text())
        definitions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                definitions.extend(
                    node for node in ast.parse("".join(cell["source"])).body
                    if isinstance(node, ast.FunctionDef) and node.name == "find_quality_issues"
                )
        self.assertTrue(definitions, "Quote-quality checks are not implemented")
        namespace = {"pd": pd, "np": np}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook_path), "exec"), namespace)
        self.check = namespace["find_quality_issues"]
        self.sessions = pd.DatetimeIndex(["2025-01-02", "2025-01-03"])
        self.valid = pd.DataFrame({
            "Date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "Ticker": ["AAPL", "AAPL"],
            "Open": [100.0, 101.0], "High": [103.0, 104.0],
            "Low": [99.0, 100.0], "Close": [102.0, 103.0],
            "Adj Close": [98.0, 99.0], "Volume": [1000.0, 1200.0],
        })

    def test_valid_quotes_and_zero_volume_are_not_discarded(self):
        self.valid.loc[0, "Volume"] = 0
        self.assertFalse(self.check(self.valid, self.sessions).any(axis=None))

    def test_missing_non_numeric_nonfinite_and_impossible_values_are_flagged(self):
        cases = [
            ("Close", np.nan), ("Close", np.inf), ("Close", 0),
            ("Adj Close", -1), ("Volume", -1), ("Volume", 1.5),
            ("High", 100), ("Low", 104), ("Open", 120),
        ]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                frame = self.valid.copy()
                frame.loc[0, column] = value
                self.assertEqual(self.check(frame, self.sessions).any(axis=1).tolist(), [True, False])
        frame = self.valid.astype({"Close": object})
        frame.loc[0, "Close"] = "bad quote"
        self.assertEqual(self.check(frame, self.sessions).any(axis=1).tolist(), [True, False])

    def test_conflicting_keys_and_non_trading_dates_are_flagged(self):
        frame = self.valid.copy()
        frame.loc[1, "Date"] = frame.loc[0, "Date"]
        self.assertTrue(self.check(frame, self.sessions).any(axis=1).all())
        frame = self.valid.copy()
        frame.loc[0, "Date"] = pd.Timestamp("2025-01-04")
        self.assertEqual(self.check(frame, self.sessions).any(axis=1).tolist(), [True, False])

    def test_price_checks_allow_small_floating_point_roundoff(self):
        frame = self.valid.copy()
        frame.loc[0, "High"] = frame.loc[0, "Close"] - 1e-10
        self.assertFalse(self.check(frame, self.sessions).any(axis=None))

    def test_coverage_window_can_start_on_a_market_holiday(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "eda.ipynb"
        notebook = json.loads(path.read_text())
        source = next(
            "".join(cell["source"]) for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "expected_grid = " in "".join(cell["source"])
        )
        fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        prices = pd.DataFrame(
            [[100, 102, 99, 101, 100, 1000], [101, 103, 100, 102, 101, 1200]],
            index=pd.to_datetime(["2010-01-04", "2010-01-08"]),
            columns=pd.MultiIndex.from_product([fields, ["AAPL"]], names=["Price", "Ticker"]),
        )
        namespace = {
            "pd": pd, "top_tickers": ["AAPL"], "ohlcv_top_not_adj": prices,
            "ohlcv_params": {"start": "2010-01-01", "end": "2010-01-09"},
        }
        exec(compile(source, str(path), "exec"), namespace)
        self.assertEqual(len(namespace["expected_grid"]), 5)
        self.assertEqual(namespace["aligned_quotes"].isna().all(axis=1).sum(), 3)


if __name__ == "__main__":
    unittest.main()

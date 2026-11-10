import ast
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "eda.ipynb"


def cache_namespace():
    notebook = json.loads(NOTEBOOK.read_text())
    cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse("".join(cell["source"]))
    # Execute the real cache helper without running the notebook's network calls.
    tree.body = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
    ]
    namespace = {}
    exec(compile(tree, str(NOTEBOOK), "exec"), namespace)
    return namespace


def unexpected_download():
    raise AssertionError("A saved dataset must be read without downloading again")


class DataCacheTests(unittest.TestCase):
    def setUp(self):
        self.namespace = cache_namespace()
        self.assertIn("load_or_download", self.namespace.keys(), "Disk caching is not implemented")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.namespace.update(DATA_DIR=self.directory, REFRESH_DATA=False)
        self.load = self.namespace["load_or_download"]
        self.frame = pd.DataFrame(
            {("Close", "AAPL"): [100.0, 101.0], ("Volume", "AAPL"): [10, 12]},
            index=pd.date_range("2025-01-02", periods=2, name="Date"),
        )
        self.frame.columns.names = ["Price", "Ticker"]

    def test_saved_data_survives_a_new_kernel_namespace(self):
        first = self.load("prices", {"start": "2025-01-01"}, lambda: self.frame.copy())
        first.iloc[0, 0] = -1
        fresh = cache_namespace()
        fresh.update(DATA_DIR=self.directory, REFRESH_DATA=False)
        cached = fresh["load_or_download"](
            "prices", {"start": "2025-01-01"}, unexpected_download
        )
        pd.testing.assert_frame_equal(cached, self.frame)

    def test_changed_request_parameters_do_not_reuse_another_dataset(self):
        parameters = {
            "tickers": ["AAPL"], "start": "2025-01-01", "auto_adjust": True,
        }
        self.load("prices", parameters, lambda: self.frame)
        changed_frame = self.frame * 2
        for change in [
            {"tickers": ["MSFT"]}, {"start": "2024-01-01"}, {"auto_adjust": False},
        ]:
            with self.subTest(change=change):
                changed = {**parameters, **change}
                result = self.load("prices", changed, lambda: changed_frame)
                pd.testing.assert_frame_equal(result, changed_frame)
                reordered = dict(reversed(list(changed.items())))
                cached = self.load("prices", reordered, unexpected_download)
                pd.testing.assert_frame_equal(cached, changed_frame)

    def test_forced_refresh_replaces_the_saved_data(self):
        self.load("prices", {}, lambda: self.frame)
        updated = self.frame * 2
        self.namespace["REFRESH_DATA"] = True
        self.load("prices", {}, lambda: updated)
        self.namespace["REFRESH_DATA"] = False
        cached = self.load("prices", {}, unexpected_download)
        pd.testing.assert_frame_equal(cached, updated)

    def test_failed_refresh_keeps_the_previous_snapshot(self):
        self.load("prices", {}, lambda: self.frame)
        self.namespace["REFRESH_DATA"] = True

        def unavailable():
            raise ConnectionError("Yahoo is unavailable")

        with self.assertRaises(ConnectionError):
            self.load("prices", {}, unavailable)
        self.namespace["REFRESH_DATA"] = False
        cached = self.load("prices", {}, unexpected_download)
        pd.testing.assert_frame_equal(cached, self.frame)

    def test_failed_or_empty_download_is_not_saved(self):
        for invalid in [None, pd.DataFrame(), pd.DataFrame({"Close": [float("nan")]})]:
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(RuntimeError):
                    self.load("prices", {}, lambda: invalid)
                self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

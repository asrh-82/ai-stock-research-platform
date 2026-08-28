import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Utils import watchlist_utils


class WatchlistPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.watchlist_file = Path(self.temp_dir.name) / "watchlist.json"
        self.file_patch = patch.object(
            watchlist_utils,
            "WATCHLIST_FILE",
            self.watchlist_file,
        )
        self.file_patch.start()

    def tearDown(self):
        self.file_patch.stop()
        self.temp_dir.cleanup()

    def test_add_normalizes_ticker_and_prevents_duplicates(self):
        self.assertTrue(watchlist_utils.add_to_watchlist(" aapl "))
        self.assertFalse(watchlist_utils.add_to_watchlist("AAPL"))
        self.assertEqual(watchlist_utils.load_watchlist(), ["AAPL"])

    def test_remove_is_case_insensitive(self):
        watchlist_utils.save_watchlist(["AAPL", "MSFT"])

        watchlist_utils.remove_from_watchlist("aapl")

        self.assertEqual(watchlist_utils.load_watchlist(), ["MSFT"])

    def test_corrupt_file_is_backed_up_and_reset(self):
        self.watchlist_file.write_text("not valid json")

        self.assertEqual(watchlist_utils.load_watchlist(), [])
        self.assertEqual(self.watchlist_file.read_text(), "[]\n")
        self.assertTrue(self.watchlist_file.with_suffix(".json.broken").exists())


if __name__ == "__main__":
    unittest.main()

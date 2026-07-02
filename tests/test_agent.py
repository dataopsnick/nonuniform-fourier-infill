import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from nufi.agent import impute_dataframe, plot_diagnostics, TransformationTracker, TransformationLoggingError

class TestAgentNativeLayer(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock dataframe with non-uniform timestamps and NaNs
        self.timestamps = np.array([1.2, 2.5, 3.1, 4.8, 5.5, 6.9, 8.1, 9.4])
        self.data_clean = 3.0 * np.sin(2.0 * np.pi * 0.1 * self.timestamps)
        
        self.data_with_nan = self.data_clean.copy()
        self.data_with_nan[2] = np.nan
        self.data_with_nan[5] = np.nan
        
        self.df = pd.DataFrame({
            "timestamp": self.timestamps,
            "signal": self.data_with_nan
        })
        
        # Paths for testing — use unique temp directories for isolation
        self._tmpdir = tempfile.mkdtemp(prefix="nufi_test_")
        self.test_log = os.path.join(self._tmpdir, "transformations.log")
        self.test_history = os.path.join(self._tmpdir, "history")

    def tearDown(self):
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_impute_dataframe_zero_config(self):
        # Test zero-config impute_dataframe
        infilled_df, diagnostics = impute_dataframe(
            self.df,
            time_col="timestamp",
            log_path=self.test_log,
            history_dir=self.test_history
        )
        
        # Verify infilling success
        self.assertFalse(infilled_df["signal"].isna().any())
        self.assertEqual(len(infilled_df), len(self.df))
        
        # Verify diagnostics metadata
        self.assertIn("signal", diagnostics)
        col_diag = diagnostics["signal"]
        self.assertIn("snr_db", col_diag)
        self.assertIn("spectral_entropy", col_diag)
        self.assertIn("stability_flags", col_diag)
        self.assertIn("optimized_alpha", col_diag)
        self.assertIn("n_frequencies", col_diag)
        
        # Verify logging & snapshots occurred
        self.assertTrue(os.path.exists(self.test_log))
        self.assertTrue(os.path.exists(self.test_history))
        
        # Count snapshot files (.csv)
        files = os.listdir(self.test_history)
        csv_files = [f for f in files if f.endswith(".csv")]
        self.assertGreaterEqual(len(csv_files), 2, f"Expected at least 2 CSV snapshots, got: {csv_files}")

    def test_impute_dataframe_empty(self):
        """Edge case: empty DataFrame should raise ValueError."""
        empty_df = pd.DataFrame(columns=["timestamp", "signal"])
        with self.assertRaises(ValueError):
            impute_dataframe(empty_df, time_col="timestamp")

    def test_impute_dataframe_all_nan(self):
        """Edge case: column with all NaN values."""
        all_nan_df = self.df.copy()
        all_nan_df["signal"] = np.nan
        result_df, diagnostics = impute_dataframe(
            all_nan_df,
            time_col="timestamp",
            log_path=self.test_log,
            history_dir=self.test_history
        )
        self.assertTrue(result_df["signal"].isna().all())
        self.assertIn("signal", diagnostics)
        col_diag = diagnostics["signal"]
        self.assertIn("stability_flags", col_diag)
        flags = col_diag["stability_flags"]
        self.assertIsInstance(flags, list)
        self.assertIn("NO_OBSERVATIONS", flags)
        # Verify other diagnostic fields have safe sentinel values
        self.assertIn("snr_db", col_diag)
        self.assertIn("spectral_entropy", col_diag)
        self.assertIn("optimized_alpha", col_diag)
        self.assertIn("n_frequencies", col_diag)

    def test_impute_dataframe_no_nans(self):
        """Edge case: DataFrame with no missing values."""
        clean_df = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0], "signal": [10.0, 20.0, 30.0]})
        result_df, _ = impute_dataframe(
            clean_df,
            time_col="timestamp",
            log_path=self.test_log,
            history_dir=self.test_history
        )
        pd.testing.assert_frame_equal(clean_df, result_df, atol=1e-8)

    def test_impute_dataframe_missing_time_col(self):
        """Edge case: specified time column does not exist."""
        with self.assertRaises(KeyError):
            impute_dataframe(self.df, time_col="nonexistent")

    def test_impute_dataframe_non_numeric_index(self):
        """Edge case: specified time column is not numeric / index is not numeric."""
        bad_df = pd.DataFrame({"signal": [1.0, 2.0]}, index=["a", "b"])
        with self.assertRaises(TypeError):
            impute_dataframe(bad_df, time_col=None)

    def test_impute_dataframe_non_dataframe_input(self):
        """Edge case: non-DataFrame input should raise TypeError."""
        for invalid in [{"timestamp": [1, 2]}, [1, 2, 3], None]:
            with self.assertRaises(TypeError):
                impute_dataframe(invalid)

    def test_tracker_logging(self):
        tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
        test_entry = {"event": "test_ping", "value": 42}
        tracker.log_transformation(test_entry)
        
        self.assertTrue(os.path.exists(self.test_log))
        with open(self.test_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        logged_data = json.loads(lines[0])
        self.assertEqual(logged_data["event"], "test_ping")
        self.assertEqual(logged_data["value"], 42)

    def test_tracker_error_enforcement(self):
        # Force a logging error by mocking open to raise an OSError
        tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
        with patch("builtins.open", mock_open()) as mocked_file:
            mocked_file.side_effect = OSError("Access Denied")
            
            with self.assertRaises(TransformationLoggingError):
                tracker.log_transformation({"event": "should_fail"})

    def test_tracker_version_reversion(self):
        tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
        
        # Save a snapshot of the original dataframe
        df_orig = self.df.copy()
        ver_id = tracker.save_snapshot(df_orig, "step_1_orig")
        
        # Mutate the dataframe
        df_mutated = df_orig.copy()
        df_mutated["signal"] = 999.0
        
        # Verify reversion returns exactly the original data
        df_reverted = tracker.revert_to_version(ver_id)
        pd.testing.assert_frame_equal(df_orig, df_reverted)
        # Guard against false positive: ensure reverted is not the mutated version
        self.assertFalse(
            df_reverted["signal"].equals(df_mutated["signal"]),
            "revert_to_version returned the mutated dataframe instead of the original snapshot"
        )

    def test_agent_plot_diagnostics(self):
        # Ensure non-interactive backend for headless CI environments
        import matplotlib
        matplotlib.use('Agg')
        
        # Run infilling
        infilled_df, diagnostics = impute_dataframe(
            self.df,
            time_col="timestamp",
            log_path=self.test_log,
            history_dir=self.test_history
        )
        
        # Test plot rendering with show_plot=False to avoid blocking tests
        save_img = os.path.join(self._tmpdir, "test_diagnostics_plot.png")
        if os.path.exists(save_img):
            os.remove(save_img)
            
        try:
            plot_diagnostics(
                original_df=self.df,
                infilled_df=infilled_df,
                diagnostics=diagnostics,
                time_col="timestamp",
                save_path=save_img,
                show_plot=False
            )
            
            # Verify the plot image was successfully created on disk
            self.assertTrue(os.path.exists(save_img))
        finally:
            if os.path.exists(save_img):
                os.remove(save_img)

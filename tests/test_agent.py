import json
import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from nufi.agent import (
    impute_dataframe,
    plot_diagnostics,
    run_tipping_point_analysis,
    plot_tipping_point,
    TransformationTracker,
    TransformationLoggingError,
)

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

        # tempfile.mkdtemp() returns a path outside the CWD on macOS
        # (/var/folders/... symlinked to /private/var/folders/...). The tracker's
        # containment guard remains a safety default for agent-driven relative
        # paths, but these tests explicitly request tempdir paths, so opt out
        # via the documented override (Task 54).
        self._prev_unsafe_paths = os.environ.get("NUFI_ALLOW_UNSAFE_PATHS")
        os.environ["NUFI_ALLOW_UNSAFE_PATHS"] = "1"

    def tearDown(self):
        if self._prev_unsafe_paths is None:
            os.environ.pop("NUFI_ALLOW_UNSAFE_PATHS", None)
        else:
            os.environ["NUFI_ALLOW_UNSAFE_PATHS"] = self._prev_unsafe_paths
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
        # Verify timestamp column is preserved unchanged
        np.testing.assert_array_equal(
            result_df["timestamp"].values,
            all_nan_df["timestamp"].values
        )
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

    def test_tracker_rejects_outside_cwd_by_default(self):
        """The containment guard must reject outside-CWD paths unless opted out (Task 54)."""
        guard_dir = tempfile.mkdtemp(prefix="nufi_guard_")
        try:
            with patch.dict(os.environ, {"NUFI_ALLOW_UNSAFE_PATHS": ""}):
                with self.assertRaises(ValueError):
                    TransformationTracker(
                        log_path=os.path.join(guard_dir, "transformations.log"),
                        history_dir=os.path.join(guard_dir, "history"),
                    )
        finally:
            shutil.rmtree(guard_dir, ignore_errors=True)

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

    def test_tracker_revert_to_version_invalid_id(self):
        """Edge case: revert_to_version with non-existent ID should raise an error."""
        tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
        with self.assertRaises((KeyError, FileNotFoundError, ValueError)):
            tracker.revert_to_version("nonexistent_id_12345")

    def test_tracker_revert_to_version_malformed_id(self):
        """Edge case: revert_to_version with empty or wrong-type IDs."""
        tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
        for bad_id in ["", 42, None]:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises((KeyError, FileNotFoundError, TypeError, ValueError)):
                    tracker.revert_to_version(bad_id)

    def test_agent_plot_diagnostics(self):
        # Ensure non-interactive backend for headless CI environments
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            self.skipTest("matplotlib not available")
        
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

    def test_plot_diagnostics_bad_input(self):
        """Edge case: plot_diagnostics with malformed diagnostics dict."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            self.skipTest("matplotlib not available")
        clean_df = pd.DataFrame({"timestamp": [1.0, 2.0], "signal": [10.0, 20.0]})
        save_img = os.path.join(self._tmpdir, "bad_diag_plot.png")
        bad_diagnostics_cases = [
            {},                                    # empty dict
            {"signal": {}},                        # missing all expected keys
            {"signal": {"snr_db": "not_a_number"}}, # wrong type
            {"signal": None},                      # None column diag
        ]
        for bad_diag in bad_diagnostics_cases:
            with self.subTest(bad_diag=bad_diag):
                try:
                     plot_diagnostics(
                         original_df=clean_df,
                         infilled_df=clean_df,
                         diagnostics=bad_diag,
                         time_col="timestamp",
                         save_path=save_img,
                         show_plot=False
                     )
                except (KeyError, TypeError, ValueError):
                     pass  # expected for truly malformed input
        if os.path.exists(save_img):
            os.remove(save_img)

    def _make_trial_df(self):
        rng = np.random.RandomState(0)
        rows = []
        for arm, shift in (("Control", 0.0), ("Treatment", 1.0)):
            for s in range(6):
                subj = f"{arm}_{s}"
                traj = shift + 0.1 * np.arange(6) + rng.normal(0, 0.05, 6)
                for i in range(6):
                    rows.append({
                        "arm": arm,
                        "subject": subj,
                        "time": float(i),
                        "outcome": traj[i] if i < 4 else np.nan,
                    })
        return pd.DataFrame(rows)

    def test_mnar_dropout_diagnostic_flags(self):
        t = np.arange(16, dtype=float)
        y = np.zeros(16, dtype=float)
        y[3], y[4] = -5.0, -12.0
        y[10], y[11] = -5.0, -12.0
        y_nan = y.copy()
        y_nan[5] = np.nan
        y_nan[12] = np.nan
        df = pd.DataFrame({"timestamp": t, "signal": y_nan})
        _, diag = impute_dataframe(
            df, time_col="timestamp",
            log_path=self.test_log, history_dir=self.test_history,
        )
        flags = diag["signal"]["stability_flags"]
        self.assertIn("MNAR_DROPOUT_LEVEL_BIAS", flags)

    def test_tipping_point_analysis_and_plot(self):
        import nufi.kernels.torch_kernels as tk
        df = self._make_trial_df()
        original = tk.solve_tikhonov_nudft
        counter = {"n": 0}

        def counting(*args, **kwargs):
            counter["n"] += 1
            return original(*args, **kwargs)

        with patch("nufi.kernels.torch_kernels.solve_tikhonov_nudft", side_effect=counting):
            res_small = run_tipping_point_analysis(
                df, arm_col="arm", outcome_col="outcome",
                delta_grid=np.array([-3.0, 0.0]), time_col="time",
                subject_col="subject", n_imputations=6, random_state=42,
                log_path=self.test_log, history_dir=self.test_history,
            )
            small_calls = counter["n"]
            counter["n"] = 0
            res_big = run_tipping_point_analysis(
                df, arm_col="arm", outcome_col="outcome",
                delta_grid=(np.array([-1.0, 0.0]), np.linspace(-4.0, 0.0, 9)),
                time_col="time", subject_col="subject", n_imputations=6, random_state=42,
                log_path=self.test_log, history_dir=self.test_history,
            )
            big_calls = counter["n"]

        self.assertEqual(small_calls, big_calls)
        # One transform per arm (one outcome column each): the count must be
        # non-zero AND independent of the delta-grid size (2x2 vs 2x9).
        self.assertGreaterEqual(small_calls, 2)
        self.assertLessEqual(big_calls, 8)
        self.assertEqual(res_big["treatment_effects"].shape, (2, 9))
        self.assertIsNotNone(res_big["tipping_point"])

        try:
            import matplotlib
            matplotlib.use("Agg")
            save_img = os.path.join(self._tmpdir, "tipping_point.png")
            plot_tipping_point(res_big, save_path=save_img, show_plot=False)
            self.assertTrue(os.path.exists(save_img))
        except ImportError:
            pass

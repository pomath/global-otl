"""End-to-end verification of every pipeline phase.

Tests are ordered by phase and must run sequentially (later phases depend on earlier ones).
Use: pytest tests/test_pipeline_phases.py -v -x
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import STNM, LAT, LON, ALT, X0, Y0, Z0


# ---------------------------------------------------------------------------
# Phase 1: TDP Loading
# ---------------------------------------------------------------------------

class TestTDPLoading:
    def test_load_tdp(self, tdp_dir):
        from gotl.io.tdp import load_tdp
        df = load_tdp(STNM, tdp_dir)
        assert set(df.columns) == {"t", "X", "Y", "Z"}
        assert len(df) > 0
        assert df["t"].dtype == "datetime64[ns]"
        # Timestamps should be in 2010
        assert df["t"].min().year == 2010
        print(f"  TDP: {len(df)} rows, {df['t'].min()} to {df['t'].max()}")

    def test_tdp_coordinates_reasonable(self, tdp_dir):
        from gotl.io.tdp import load_tdp
        df = load_tdp(STNM, tdp_dir)
        # ECEF coordinates should be near the reference position
        from conftest import X0, Y0, Z0
        assert abs(df["X"].median() - X0) < 1.0  # within 1 m
        assert abs(df["Y"].median() - Y0) < 1.0
        assert abs(df["Z"].median() - Z0) < 1.0


# ---------------------------------------------------------------------------
# Phase 2: OTL Params Loading
# ---------------------------------------------------------------------------

class TestOTLParams:
    def test_read_otl_params_default(self, otl_params_dir):
        from gotl.io.otl_params import read_otl_params
        df = read_otl_params(STNM, otl_params_dir)
        assert df.shape == (11, 6)
        assert list(df.columns) == ["dU", "dW", "dS", "gU", "gW", "gS"]
        from gotl import CONSTITUENTS
        assert list(df.index) == CONSTITUENTS
        # M2 radial amplitude should be ~10 mm
        assert 0.005 < df.loc["M2", "dU"] < 0.020
        print(f"  M2 dU = {df.loc['M2', 'dU']*1e3:.2f} mm")

    def test_read_otl_params_with_suffix(self, otl_params_dir):
        from gotl.io.otl_params import read_otl_params
        df = read_otl_params(STNM, otl_params_dir, suffix="_FES2014b")
        assert df.shape == (11, 6)
        assert df.loc["M2", "dU"] > 0


# ---------------------------------------------------------------------------
# Phase 3: Hardisp Generation + Loading
# ---------------------------------------------------------------------------

class TestHardisp:
    def test_hardisp_binary_runs(self, otl_params_dir):
        """Run hardisp for 1 day and verify output format."""
        from gotl.hardisp_gen.runner import run_hardisp_day
        exe = Path(__file__).resolve().parent.parent / "hardisp" / "hardisp_exe"
        otl_file = otl_params_dir / f"{STNM}_FES2014b.otl"
        if not exe.exists():
            pytest.skip("hardisp_exe not compiled")

        stdout = run_hardisp_day(exe, otl_file, 2010, 1)
        lines = stdout.strip().split("\n")
        assert len(lines) == 288, f"Expected 288 lines, got {len(lines)}"
        # Raw hardisp output: 3 columns (dU dS dW in metres)
        cols = lines[0].split()
        assert len(cols) == 3, f"Expected 3 columns, got {len(cols)}"
        # dU should be in metres, mm-level
        dU = float(cols[0])
        assert abs(dU) < 0.1, f"dU={dU} seems too large"
        print(f"  DOY 1 sample: dU={float(cols[0])*1e3:.2f} dS={float(cols[1])*1e3:.2f} dW={float(cols[2])*1e3:.2f} mm")

    def test_hardisp_generate_year(self, hardisp_dir, otl_params_dir):
        """Generate a full year and verify file exists."""
        stn_dir = hardisp_dir / STNM
        outfile = stn_dir / f"{STNM}-hardisp_2010.txt"
        if not outfile.exists():
            pytest.skip("hardisp not generated (binary missing?)")
        text = outfile.read_text()
        lines = [l for l in text.strip().split("\n") if l.strip()]
        # 365 days * 288 = 105120 rows for 2010 (not leap year)
        expected = 365 * 288
        assert len(lines) == expected, f"Expected {expected}, got {len(lines)}"
        # Verify 8-column format (year doy h m s dU dS dW)
        cols = lines[0].split()
        assert len(cols) == 8, f"Expected 8 columns, got {len(cols)}: {lines[0]}"

    def test_load_hardisp(self, hardisp_dir):
        """Load the generated hardisp files."""
        stn_dir = hardisp_dir / STNM
        if not (stn_dir / f"{STNM}-hardisp_2010.txt").exists():
            pytest.skip("hardisp not generated")

        from gotl.io.hardisp import load_hardisp
        df = load_hardisp(STNM, hardisp_dir, years=["2010"])
        assert set(df.columns) == {"t", "dU", "dS", "dW"}
        assert len(df) == 365 * 288
        assert df["t"].min().year == 2010
        # Displacements should be mm-level
        assert df["dU"].abs().max() < 0.1
        print(f"  Hardisp: {len(df)} rows, dU range [{df['dU'].min()*1e3:.2f}, {df['dU'].max()*1e3:.2f}] mm")


# ---------------------------------------------------------------------------
# Phase 4: SLTM Fit
# ---------------------------------------------------------------------------

class TestSLTM:
    def test_design_matrix(self):
        from gotl.sltm.design import design_sltm
        t = np.linspace(2010.0, 2010.5, 1000)
        A = design_sltm(t, tref=2010.25, n=1, tjmp=None, fperiods=np.array([1.0, 0.5]))
        # n=1 → 2 trend cols; 0 jumps; 2 periods × 2 = 4 osc cols → 6 total
        assert A.shape == (1000, 6)

    def test_design_with_jumps(self):
        from gotl.sltm.design import design_sltm
        t = np.linspace(2010.0, 2011.0, 1000)
        tjmp = np.array([2010.5])
        A = design_sltm(t, tref=2010.5, n=1, tjmp=tjmp, fperiods=np.array([1.0]))
        # 2 trend + 1 jump + 2 osc = 5
        assert A.shape == (1000, 5)

    def test_irwls_recovers_trend(self):
        """IRWLS should recover a known linear trend."""
        from gotl.sltm.solver import irwls
        from gotl.sltm.design import design_sltm
        rng = np.random.default_rng(123)
        t = np.linspace(2010.0, 2011.0, 5000)
        A = design_sltm(t, tref=2010.5, n=1)
        # True model: intercept=1.0, velocity=0.05
        m_true = np.array([1.0, 0.05])
        b = A @ m_true + rng.normal(0, 0.01, len(t))
        result = irwls(A, b, sig=0.01)
        assert abs(result.x[0] - 1.0) < 0.01
        assert abs(result.x[1] - 0.05) < 0.01
        print(f"  IRWLS: intercept={result.x[0]:.4f} (true 1.0), velocity={result.x[1]:.4f} (true 0.05)")

    def test_fit_sltm_3component(self):
        """fit_sltm on 3-component synthetic data."""
        from gotl.sltm.fit import fit_sltm
        rng = np.random.default_rng(456)
        t = np.linspace(2010.0, 2011.0, 5000)
        nt = len(t)
        # Known trend per component
        pvs = np.zeros((3, nt))
        pvs[0, :] = 0.1 + 0.02 * (t - 2010.5) + rng.normal(0, 0.005, nt)
        pvs[1, :] = -0.05 + 0.01 * (t - 2010.5) + rng.normal(0, 0.005, nt)
        pvs[2, :] = 0.3 - 0.03 * (t - 2010.5) + rng.normal(0, 0.005, nt)

        fit = fit_sltm(t, tref=2010.5, pvs=pvs, sig=0.005, n=1)
        assert fit.cpvs.shape == (3, nt)
        assert fit.swts.shape == (3, nt)
        assert fit.pout.shape == (3,)
        # Residuals should be small
        for i in range(3):
            residual_rms = np.sqrt(np.mean((pvs[i, :] - fit.cpvs[i, :]) ** 2))
            assert residual_rms < 0.02, f"Component {i}: residual RMS = {residual_rms}"
        rms = [np.sqrt(np.mean((pvs[i, :] - fit.cpvs[i, :]) ** 2)) for i in range(3)]
        print(f"  fit_sltm: residual RMS = {rms}")


# ---------------------------------------------------------------------------
# Phase 5: Cache Round-trip
# ---------------------------------------------------------------------------

class TestCache:
    def test_tdp_roundtrip(self, results_dir):
        from gotl.io.cache import StationCache
        cache = StationCache(results_dir / "test_cache.h5")
        t = pd.date_range("2010-01-01", periods=100, freq="5min")
        df = pd.DataFrame({"t": t, "X": np.ones(100), "Y": np.ones(100) * 2, "Z": np.ones(100) * 3})
        cache.write_tdp(df)
        assert cache.has("tdp")
        df2 = cache.read_tdp()
        assert len(df2) == 100
        # Timestamps should survive roundtrip (within 1 second precision due to float64)
        dt_diff = (df2["t"] - df["t"]).abs().max()
        assert dt_diff < pd.Timedelta(seconds=1)

    def test_hardisp_roundtrip(self, results_dir):
        from gotl.io.cache import StationCache
        cache = StationCache(results_dir / "test_cache.h5")
        t = pd.date_range("2010-01-01", periods=100, freq="5min")
        df = pd.DataFrame({"t": t, "dU": np.ones(100) * 0.01, "dS": np.zeros(100), "dW": np.zeros(100)})
        cache.write_hardisp(df)
        assert cache.has("hardisp")
        df2 = cache.read_hardisp()
        np.testing.assert_allclose(df2["dU"].values, 0.01, atol=1e-10)

    def test_combined_roundtrip(self, results_dir):
        from gotl.io.cache import StationCache
        from gotl.registry import StationMeta
        cache = StationCache(results_dir / "test_cache.h5")
        t = pd.date_range("2010-01-01", periods=100, freq="5min")
        df = pd.DataFrame({
            "t": t,
            "u_nop": np.ones(100) * 0.001,
            "s_nop": np.ones(100) * 0.002,
            "w_nop": np.ones(100) * 0.003,
            "dU": np.ones(100) * 0.01,
            "dS": np.zeros(100),
            "dW": np.zeros(100),
        })
        meta = StationMeta(stnm=STNM, lat=LAT, lon=LON, alt_m=ALT,
                           network="POLENET",
                           data_start=pd.Timestamp("2010-01-01"),
                           data_end=pd.Timestamp("2010-12-31"))
        cache.write_combined(df, meta)
        assert cache.has("combined")
        df2 = cache.read_combined()
        assert set(df2.columns) >= {"t", "u_nop", "s_nop", "w_nop", "dU", "dS", "dW"}
        np.testing.assert_allclose(df2["u_nop"].values, 0.001, atol=1e-10)

    def test_otl_coeff_roundtrip(self, results_dir):
        from gotl.io.cache import StationCache
        from gotl import CONSTITUENTS
        cache = StationCache(results_dir / "test_cache.h5")
        rng = np.random.default_rng(789)
        df = pd.DataFrame({
            "dU": rng.uniform(0.001, 0.02, 11),
            "dS": rng.uniform(0.0001, 0.005, 11),
            "dW": rng.uniform(0.0001, 0.005, 11),
            "gU": rng.uniform(-180, 180, 11),
            "gS": rng.uniform(-180, 180, 11),
            "gW": rng.uniform(-180, 180, 11),
            "sdU": rng.uniform(0.0001, 0.001, 11),
            "sdS": rng.uniform(0.0001, 0.001, 11),
            "sdW": rng.uniform(0.0001, 0.001, 11),
            "sgU": rng.uniform(0.1, 5, 11),
            "sgS": rng.uniform(0.1, 5, 11),
            "sgW": rng.uniform(0.1, 5, 11),
        }, index=CONSTITUENTS)
        cache.write_otl_coeff(df)
        assert cache.has("otl_coeff")
        df2 = cache.read_otl_coeff()
        assert "sdU" in df2.columns, "Uncertainty columns should be preserved"
        np.testing.assert_allclose(df2["dU"].values, df["dU"].values, atol=1e-10)
        np.testing.assert_allclose(df2["sdU"].values, df["sdU"].values, atol=1e-10)


# ---------------------------------------------------------------------------
# Phase 6: Coordinate Transforms
# ---------------------------------------------------------------------------

class TestCoords:
    def test_ecef_to_lla_roundtrip(self):
        from gotl.coords.transforms import ecef_to_lla
        from conftest import X0, Y0, Z0
        lat, lon, alt = ecef_to_lla(X0, Y0, Z0)
        assert abs(lat - LAT) < 0.001
        assert abs(lon - LON) < 0.001
        assert abs(alt - ALT) < 1.0

    def test_ecef_to_enu(self):
        from gotl.coords.transforms import ecef_to_lla, ecef_to_enu
        from conftest import X0, Y0, Z0
        lat0, lon0, alt0 = ecef_to_lla(X0, Y0, Z0)
        e, n, u = ecef_to_enu(X0, Y0, Z0, lat0, lon0, alt0)
        # At the reference point, ENU should be (0, 0, 0)
        assert abs(e) < 0.001
        assert abs(n) < 0.001
        assert abs(u) < 0.001


# ---------------------------------------------------------------------------
# Phase 6b: Outlier masking before VTide
# ---------------------------------------------------------------------------

class TestOutlierMasking:
    def _build_df(self, n):
        t = pd.date_range("2010-01-01", periods=n, freq="5min")
        return pd.DataFrame({
            "t": t,
            "u_nop": np.full(n, 0.001),
            "s_nop": np.full(n, 0.002),
            "w_nop": np.full(n, 0.003),
            "dU": np.zeros(n),
            "dS": np.zeros(n),
            "dW": np.zeros(n),
        })

    def test_mask_outliers_drops_residual_spikes(self):
        """Spikes on any component should be dropped via the k·σ_robust rule."""
        from gotl.pipeline import _mask_outliers
        n = 100
        df = self._build_df(n)
        # Build cm-scale Gaussian noise on each component so σ_robust > floor.
        rng = np.random.default_rng(0)
        df["u_nop"] = rng.normal(0, 0.02, n)
        df["s_nop"] = rng.normal(0, 0.02, n)
        df["w_nop"] = rng.normal(0, 0.02, n)
        # Inject 1-m spikes on different components at five distinct rows.
        bad = [10, 25, 40, 55, 70]
        df.loc[bad[0], "u_nop"] = 1.0
        df.loc[bad[1], "s_nop"] = 1.0
        df.loc[bad[2], "w_nop"] = 1.0
        df.loc[bad[3], "u_nop"] = -1.0
        df.loc[bad[4], "s_nop"] = -1.0

        df_clean, stats = _mask_outliers(df)
        assert stats["n_total"] == n
        assert stats["n_dropped"] == len(bad)
        assert len(df_clean) == n - len(bad)
        kept_times = set(df_clean["t"].astype("int64"))
        for i in bad:
            assert int(df["t"].iloc[i].value) not in kept_times

    def test_mask_outliers_stats_fields(self):
        """Stats fields are populated; cm-noise + m-spikes triggers drops via σ_floor."""
        from gotl.pipeline import _mask_outliers
        n = 50
        df = self._build_df(n)
        df.loc[3, "u_nop"] = 15.0      # 15-m spike
        df.loc[7, "w_nop"] = -9.0
        df_clean, stats = _mask_outliers(df)
        assert set(stats) == {"n_total", "n_dropped",
                               "max_abs_u", "max_abs_s", "max_abs_w"}
        assert stats["max_abs_u"] == pytest.approx(15.0)
        assert stats["max_abs_w"] == pytest.approx(9.0)
        # The two m-scale spikes blow past 4.5·σ_floor (4.5 cm); both are dropped.
        assert stats["n_dropped"] == 2
        assert len(df_clean) == n - 2

    def test_combined_clean_roundtrip_with_stats(self, results_dir):
        from gotl.io.cache import StationCache
        from gotl.registry import StationMeta
        cache = StationCache(results_dir / "outlier_cache.h5")
        df = self._build_df(20)
        meta = StationMeta(stnm=STNM, lat=LAT, lon=LON, alt_m=ALT,
                           network="POLENET",
                           data_start=pd.Timestamp("2010-01-01"),
                           data_end=pd.Timestamp("2010-12-31"))
        stats = {"n_total": 25, "n_dropped": 5,
                 "max_abs_u": 12.5, "max_abs_s": 0.003, "max_abs_w": 0.004}
        cache.write_combined(df, meta, stats=stats)
        cache.write_combined_clean(df.iloc[:15], meta)

        assert cache.has("combined")
        assert cache.has("combined_clean")
        attrs = cache.read_combined_attrs()
        assert attrs["n_total"] == 25
        assert attrs["n_dropped"] == 5
        assert attrs["max_abs_u"] == pytest.approx(12.5)
        assert attrs["stnm"] == STNM
        df_clean = cache.read_combined_clean()
        assert len(df_clean) == 15

    def test_combine_and_fit_flags_injected_spikes(self):
        """Synthetic 2-component trend + random noise + 5 large spikes."""
        from gotl.pipeline import _combine_and_fit, _mask_outliers
        from gotl.registry import StationMeta

        rng = np.random.default_rng(2026)
        n = 2000
        t_index = pd.date_range("2010-01-01", periods=n, freq="5min")
        # TDP jitters ±2 mm around reference ECEF
        noise = 0.002
        X = X0 + rng.normal(0, noise, n)
        Y = Y0 + rng.normal(0, noise, n)
        Z = Z0 + rng.normal(0, noise, n)
        # Inject 10-m spikes on X at 5 distinct indices.
        bad = [100, 500, 900, 1400, 1800]
        for i in bad:
            X[i] += 10.0

        df_tdp = pd.DataFrame({"t": t_index, "X": X, "Y": Y, "Z": Z})
        df_hardisp = pd.DataFrame({
            "t": t_index,
            "dU": np.zeros(n),
            "dS": np.zeros(n),
            "dW": np.zeros(n),
        })
        meta = StationMeta(stnm=STNM, lat=LAT, lon=LON, alt_m=ALT,
                           network="POLENET",
                           data_start=pd.Timestamp("2010-01-01"),
                           data_end=pd.Timestamp("2010-12-31"))

        df_combined, _swts = _combine_and_fit(df_tdp, df_hardisp, meta)
        df_clean, stats = _mask_outliers(df_combined)
        # All 5 spikes should be dropped (10-m on X projects onto E and U).
        kept_times = set(df_clean["t"].astype("int64"))
        for i in bad:
            assert int(df_combined["t"].iloc[i].value) not in kept_times, \
                f"spike at {i} was not dropped"
        assert stats["n_total"] == n
        assert stats["n_dropped"] >= len(bad)
        # The combined residuals should show the ~10 m excursion before masking.
        assert max(stats["max_abs_u"], stats["max_abs_w"]) > 1.0
        # After masking, no component residual should exceed ~0.1 m.
        for col in ("u_nop", "s_nop", "w_nop"):
            assert df_clean[col].abs().max() < 0.1


# ---------------------------------------------------------------------------
# Phase 7: Compare / Vector Misfit
# ---------------------------------------------------------------------------

class TestCompare:
    def test_vector_misfit_identical(self):
        from gotl.compare.residuals import _vector_misfit
        assert _vector_misfit(0.01, 45.0, 0.01, 45.0) == pytest.approx(0.0, abs=1e-12)

    def test_vector_misfit_phase_180(self):
        from gotl.compare.residuals import _vector_misfit
        # 180 degree phase difference → misfit = 2 * amplitude
        vmfit = _vector_misfit(0.01, 0.0, 0.01, 180.0)
        assert vmfit == pytest.approx(0.02, abs=1e-10)

    def test_vector_misfit_amplitude_only(self):
        from gotl.compare.residuals import _vector_misfit
        # Same phase, different amplitude
        vmfit = _vector_misfit(0.01, 45.0, 0.005, 45.0)
        assert vmfit == pytest.approx(0.005, abs=1e-10)

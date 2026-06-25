"""Tests for the cloud pipeline module.

Tests the local workflow: serialize metadata, stage files, process in
Dask LocalCluster workers, collect results. No AWS costs.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from gotl.cloud.worker import serialize_meta, deserialize_meta
from gotl.registry import StationMeta

import pandas as pd


# ── Serialization tests ──────────────────────────────────────────


class TestMetaSerialization:
    """Test StationMeta serialization for cross-process transfer."""

    def test_roundtrip_basic(self):
        meta = StationMeta(
            stnm="aboa",
            lat=-73.0435,
            lon=-13.4073,
            alt_m=468.7,
            network="POLENET",
            data_start=pd.Timestamp("2000-01-01"),
            data_end=pd.Timestamp("2022-12-31"),
            jump_epochs=[],
        )
        d = serialize_meta(meta)
        restored = deserialize_meta(d)

        assert restored.stnm == "aboa"
        assert restored.lat == -73.0435
        assert restored.lon == -13.4073
        assert restored.alt_m == 468.7
        assert restored.network == "POLENET"
        assert restored.data_start == pd.Timestamp("2000-01-01")
        assert restored.data_end == pd.Timestamp("2022-12-31")
        assert restored.jump_epochs == []

    def test_roundtrip_with_jumps(self):
        meta = StationMeta(
            stnm="brux",
            lat=50.7986,
            lon=4.3581,
            alt_m=158.0,
            network="IGS",
            data_start=pd.Timestamp("2010-01-01"),
            data_end=pd.Timestamp("2023-06-30"),
            jump_epochs=[
                pd.Timestamp("2015-03-15"),
                pd.Timestamp("2019-11-20"),
            ],
        )
        d = serialize_meta(meta)
        restored = deserialize_meta(d)

        assert len(restored.jump_epochs) == 2
        assert restored.jump_epochs[0] == pd.Timestamp("2015-03-15")
        assert restored.jump_epochs[1] == pd.Timestamp("2019-11-20")

    def test_roundtrip_none_dates(self):
        meta = StationMeta(
            stnm="test",
            lat=0.0,
            lon=0.0,
            alt_m=0.0,
            network="TEST",
            data_start=None,
            data_end=None,
            jump_epochs=[],
        )
        d = serialize_meta(meta)
        restored = deserialize_meta(d)

        assert restored.data_start is None
        assert restored.data_end is None

    def test_serialized_is_json_safe(self):
        """Serialized dict should contain only JSON-compatible types."""
        import json

        meta = StationMeta(
            stnm="aboa",
            lat=-73.0435,
            lon=-13.4073,
            alt_m=468.7,
            network="POLENET",
            data_start=pd.Timestamp("2000-01-01"),
            data_end=pd.Timestamp("2022-12-31"),
            jump_epochs=[pd.Timestamp("2010-05-15")],
        )
        d = serialize_meta(meta)
        # Should not raise
        json.dumps(d)


# ── Local staging tests ──────────────────────────────────────────


class TestLocalStaging:
    """Test file staging for cloud workers."""

    def test_stage_station_inputs(self, tdp_dir, hardisp_dir, otl_params_dir, tmp_root):
        """Staging copies the right files to the right layout."""
        from gotl.cloud.local import _stage_station_inputs
        from gotl.config import Config

        cfg = Config(
            results_dir=tmp_root / "results",
            tdp_root=tdp_dir,
            hardisp_root=hardisp_dir,
            otl_params_dir=otl_params_dir,
        )

        staging = tmp_root / "staging_test" / "aboa"
        _stage_station_inputs("aboa", cfg, staging)

        # TDP files should be in staging/tdp/aboa_results/
        tdp_staged = staging / "tdp" / "aboa_results"
        assert tdp_staged.is_dir()
        tdp_files = list(tdp_staged.glob("*.tdp"))
        assert len(tdp_files) > 0, "No TDP files staged"

        # Hardisp files should be in staging/hardisp/aboa/
        hd_staged = staging / "hardisp" / "aboa"
        if (hardisp_dir / "aboa").is_dir():
            assert hd_staged.is_dir()

        # OTL params should be in staging/otl_params/
        otl_staged = staging / "otl_params"
        assert otl_staged.is_dir()


# ── Full local workflow test ─────────────────────────────────────


class TestLocalWorkflow:
    """End-to-end local test using Dask LocalCluster."""

    def test_local_worker_processes_station(
        self, tdp_dir, hardisp_dir, otl_params_dir, tmp_root
    ):
        """Test that _local_worker can process a station in isolation."""
        from gotl.cloud.local import _local_worker, _stage_station_inputs
        from gotl.cloud.worker import serialize_meta
        from gotl.config import Config
        from gotl.registry import StationMeta

        meta = StationMeta(
            stnm="aboa",
            lat=-73.0435,
            lon=-13.4073,
            alt_m=468.7,
            network="POLENET",
            data_start=pd.Timestamp("2010-01-01"),
            data_end=pd.Timestamp("2010-01-30"),
            jump_epochs=[],
        )

        cfg = Config(
            results_dir=tmp_root / "results",
            tdp_root=tdp_dir,
            hardisp_root=hardisp_dir,
            otl_params_dir=otl_params_dir,
        )

        # Stage files
        staging = tmp_root / "worker_test" / "aboa"
        _stage_station_inputs("aboa", cfg, staging)

        # Run worker directly (not via Dask, just test the function)
        result = _local_worker(
            stnm="aboa",
            meta_dict=serialize_meta(meta),
            staging_dir=str(staging),
            tide_models=["TPXO9"],
            force=True,
        )

        assert result["stnm"] == "aboa"
        assert result["status"] == "ok", f"Worker failed: {result.get('error')}"
        assert result["elapsed_sec"] > 0

        # Check that .h5 was created
        h5_path = Path(result["h5_path"])
        assert h5_path.exists(), f"HDF5 result not found at {h5_path}"

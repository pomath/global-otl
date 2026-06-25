"""HDF5-based result cache, one file per station.

Layout of {stnm}.h5:
    /tdp/
        t          float64[N]  Unix timestamps (seconds since 1970-01-01)
        X, Y, Z    float64[N]  ECEF coordinates (m)
    /hardisp/
        t          float64[M]  Unix timestamps
        dU, dS, dW float64[M]  displacement (m)
    /combined/
        t                      float64[K]
        u_nop, s_nop, w_nop    float64[K]  SLTM residuals (Harpos convention, m)
        dU, dS, dW             float64[K]  hardisp forward model (m)
        attrs:
            lat, lon, alt_m    float64
            stnm               str
            n_total            int       total combined epochs
            n_dropped          int       epochs below IRWLS weight threshold
            max_abs_u/s/w      float64   max |residual| per component (m)
    /combined_clean/
        Same schema as /combined, but with outlier epochs removed. This is
        the series fed to VTide (which has no outlier resistance). Stations
        with no outliers have /combined_clean identical to /combined.
    /otl_coeff/
        dU, dS, dW             float64[11]  VTide amplitudes (m)
        gU, gS, gW             float64[11]  VTide phases (degrees)
        sdU, sdS, sdW          float64[11]  amplitude uncertainties (m)
        sgU, sgS, sgW          float64[11]  phase uncertainties (degrees)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def _ts_to_float(t_series: pd.Series) -> np.ndarray:
    """Convert datetime Series to Unix seconds (float64)."""
    return t_series.astype("int64").values / 1e9


def _float_to_ts(arr: np.ndarray) -> pd.Series:
    """Convert Unix seconds array to datetime Series."""
    return pd.to_datetime(arr * 1e9, unit="ns")


class StationCache:
    """Read/write HDF5 cache for one station."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def has(self, step: str) -> bool:
        """Return True if the cache group for *step* already exists."""
        if not self.path.exists():
            return False
        with h5py.File(self.path, "r") as f:
            return step in f

    def write_tdp(self, df: pd.DataFrame) -> None:
        """Write TDP DataFrame (columns: t, X, Y, Z) to /tdp group."""
        with h5py.File(self.path, "a") as f:
            g = f.require_group("tdp")
            _write_or_replace(g, "t", _ts_to_float(df["t"]))
            for col in ("X", "Y", "Z"):
                _write_or_replace(g, col, df[col].values)

    def read_tdp(self) -> pd.DataFrame:
        with h5py.File(self.path, "r") as f:
            g = f["tdp"]
            t = _float_to_ts(g["t"][:])
            return pd.DataFrame({
                "t": t, "X": g["X"][:], "Y": g["Y"][:], "Z": g["Z"][:]
            })

    def write_hardisp(self, df: pd.DataFrame) -> None:
        """Write hardisp DataFrame (columns: t, dU, dS, dW) to /hardisp group."""
        with h5py.File(self.path, "a") as f:
            g = f.require_group("hardisp")
            _write_or_replace(g, "t", _ts_to_float(df["t"]))
            for col in ("dU", "dS", "dW"):
                _write_or_replace(g, col, df[col].values)

    def read_hardisp(self) -> pd.DataFrame:
        with h5py.File(self.path, "r") as f:
            g = f["hardisp"]
            t = _float_to_ts(g["t"][:])
            return pd.DataFrame({
                "t": t, "dU": g["dU"][:], "dS": g["dS"][:], "dW": g["dW"][:]
            })

    _COMBINED_COLS = ("u_nop", "s_nop", "w_nop", "dU", "dS", "dW")

    def write_combined(
        self,
        df: pd.DataFrame,
        meta,
        stats: dict | None = None,
    ) -> None:
        """Write combined DataFrame and metadata to /combined group.

        Args:
            df: DataFrame with columns: t, u_nop, s_nop, w_nop, dU, dS, dW
            meta: StationMeta (or any object with .lat, .lon, .alt_m, .stnm)
            stats: optional dict of scalar quality metrics (n_total, n_dropped,
                   max_abs_u, max_abs_s, max_abs_w) to store as group attrs.
        """
        self._write_combined_group(df, meta, "combined", stats=stats)

    def read_combined(self) -> pd.DataFrame:
        """Return DataFrame for the combined step."""
        return self._read_combined_group("combined")

    def write_combined_clean(
        self,
        df: pd.DataFrame,
        meta,
    ) -> None:
        """Write outlier-filtered combined DataFrame to /combined_clean group.

        Same schema as /combined; this is the series fed to VTide.
        """
        self._write_combined_group(df, meta, "combined_clean")

    def read_combined_clean(self) -> pd.DataFrame:
        """Return DataFrame for the outlier-filtered combined series."""
        return self._read_combined_group("combined_clean")

    def read_combined_attrs(self) -> dict:
        """Return attrs dict on /combined (quality metrics + station metadata)."""
        with h5py.File(self.path, "r") as f:
            return {k: _unwrap_attr(v) for k, v in f["combined"].attrs.items()}

    def _write_combined_group(
        self,
        df: pd.DataFrame,
        meta,
        group_name: str,
        stats: dict | None = None,
    ) -> None:
        with h5py.File(self.path, "a") as f:
            g = f.require_group(group_name)
            _write_or_replace(g, "t", _ts_to_float(df["t"]))
            for col in self._COMBINED_COLS:
                _write_or_replace(g, col, df[col].values)
            g.attrs["lat"] = float(meta.lat)
            g.attrs["lon"] = float(meta.lon)
            g.attrs["alt_m"] = float(meta.alt_m)
            g.attrs["stnm"] = meta.stnm
            if stats:
                for k, v in stats.items():
                    g.attrs[k] = v

    def _read_combined_group(self, group_name: str) -> pd.DataFrame:
        with h5py.File(self.path, "r") as f:
            g = f[group_name]
            t = _float_to_ts(g["t"][:])
            return pd.DataFrame(
                {"t": t, **{c: g[c][:] for c in self._COMBINED_COLS}}
            )

    _OTL_COLS = ("dU", "dS", "dW", "gU", "gS", "gW",
                 "sdU", "sdS", "sdW", "sgU", "sgS", "sgW")

    def write_otl_coeff(self, df: pd.DataFrame) -> None:
        """Write OTL coefficient DataFrame (index=CONSTITUENTS, 12 columns)."""
        with h5py.File(self.path, "a") as f:
            g = f.require_group("otl_coeff")
            for col in self._OTL_COLS:
                if col in df.columns:
                    _write_or_replace(g, col, df[col].values)

    def read_otl_coeff(self) -> pd.DataFrame:
        from gotl import CONSTITUENTS
        with h5py.File(self.path, "r") as f:
            g = f["otl_coeff"]
            data = {col: g[col][:] for col in self._OTL_COLS if col in g}
        return pd.DataFrame(data, index=CONSTITUENTS)


def _write_or_replace(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write dataset, replacing if it already exists."""
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def _unwrap_attr(v):
    """Decode h5py attrs to native Python (bytes→str, 0-d arrays→scalar)."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v

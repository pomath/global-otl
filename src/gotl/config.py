"""Configuration dataclass for the global OTL pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Paths and settings for a global OTL pipeline run.

    Attributes:
        results_dir: Directory for per-station .h5 cache files.
        tdp_root:    Root directory for GipsyX TDP output.
                     Expected layout: {tdp_root}/{stnm}_results/*.tdp
        hardisp_root: Directory for generated hardisp time series files.
                     Expected layout: {hardisp_root}/{stnm}/{stnm}-hardisp_{YYYY}.txt
        otl_params_dir: Directory for per-station .db / .otl files
                     (Harpos format; one file per station per tide model).
                     Expected layout: {otl_params_dir}/{stnm}_{model}.otl
        registry_path: Path to station_registry.csv.
        hardisp_exe:  Path to compiled hardisp binary.
        tide_models:  List of tide model names to compare (e.g. ["TPXO9", "FES2014b"]).
        rinex_root:   Root directory for RINEX files (Phase 0).
                     Expected layout: {rinex_root}/{stnm}/*.Z
        gipsyx_rc:   Path to GipsyX rc_GipsyX.sh (None = already on PATH).
        jpl_products_dir: Root of JPL GNSS orbit/clock products.
                     Expected layout: {jpl_products_dir}/{YYYY}/
        vmf1_dir:    VMF1 troposphere data directory (optional).
        gipsy_max_workers: Max parallel GipsyX subprocesses.
        gipsy_rate_sec: Decimate data records to this sampling rate (seconds).
                     Default 300 (5-min) matches the kinematicPPP.tree
                     configuration. Use 1800 for 30-min sampling.
        gipsy_otl_model: Tide model name for GipsyX OTL correction (e.g. "FES2014").
                     GipsyX applies OTL during PPP using this model's .otl file
                     from otl_params_dir (resolved as {stnm}_{model}.otl). The same
                     file is used for hardisp generation so the correction can be
                     backed out exactly.
        loaddef_root: Path to LoadDef-main/ source directory.
        loaddef_data_dir: Path to LoadDef data (Green's functions, grids, mask).
        loaddef_mpi_np: Number of MPI processes for LoadDef.
    """

    results_dir: Path
    tdp_root: Path
    hardisp_root: Path
    otl_params_dir: Path
    registry_path: Path = field(default_factory=lambda: Path("data/station_registry.csv"))
    hardisp_exe: Path = field(default_factory=lambda: Path("hardisp/hardisp_exe"))
    tide_models: list[str] = field(default_factory=lambda: ["TPXO9"])
    rinex_root: Path | None = None
    gipsyx_rc: Path | None = None
    jpl_products_dir: Path | None = None
    vmf1_dir: Path | None = None
    gipsy_max_workers: int = 4
    gipsy_rate_sec: int = 300
    gipsy_otl_model: str = "FES2014"
    gipsy_skip_full_merge: bool = False
    loaddef_root: Path | None = None
    loaddef_data_dir: Path | None = None
    loaddef_mpi_np: int = 4

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.tdp_root = Path(self.tdp_root)
        self.hardisp_root = Path(self.hardisp_root)
        self.otl_params_dir = Path(self.otl_params_dir)
        self.registry_path = Path(self.registry_path)
        self.hardisp_exe = Path(self.hardisp_exe)
        if self.rinex_root is not None:
            self.rinex_root = Path(self.rinex_root)
        if self.gipsyx_rc is not None:
            self.gipsyx_rc = Path(self.gipsyx_rc)
        if self.jpl_products_dir is not None:
            self.jpl_products_dir = Path(self.jpl_products_dir)
        if self.vmf1_dir is not None:
            self.vmf1_dir = Path(self.vmf1_dir)
        if self.loaddef_root is not None:
            self.loaddef_root = Path(self.loaddef_root)
        if self.loaddef_data_dir is not None:
            self.loaddef_data_dir = Path(self.loaddef_data_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

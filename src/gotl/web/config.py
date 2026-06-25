"""Web-specific configuration, wrapping the core gotl Config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from gotl.config import Config


@dataclass
class WebConfig:
    """Configuration for the web dashboard."""

    gotl: Config
    host: str = "0.0.0.0"
    port: int = 8050
    max_concurrent_jobs: int = 2
    db_path: Path = field(default_factory=lambda: Path("jobs.db"))

    @classmethod
    def from_env(cls) -> WebConfig:
        """Build WebConfig from environment variables."""
        results_dir = os.environ.get("GOTL_RESULTS_DIR", "results")
        gotl_cfg = Config(
            results_dir=results_dir,
            tdp_root=os.environ.get("GOTL_TDP_ROOT", ""),
            hardisp_root=os.environ.get("GOTL_HARDISP_ROOT", ""),
            otl_params_dir=os.environ.get("GOTL_OTL_PARAMS_DIR", "data/otl_params"),
            registry_path=os.environ.get("GOTL_REGISTRY", "data/station_registry.csv"),
            hardisp_exe=os.environ.get("GOTL_HARDISP_EXE", "hardisp/hardisp_exe"),
            tide_models=[
                m.strip()
                for m in os.environ.get("GOTL_TIDE_MODELS", "TPXO9").split(",")
            ],
            rinex_root=os.environ.get("GOTL_RINEX_ROOT") or None,
            gipsyx_rc=os.environ.get("GOTL_GIPSYX_RC") or None,
            jpl_products_dir=os.environ.get("GOTL_JPL_PRODUCTS") or None,
            vmf1_dir=os.environ.get("GOTL_VMF1_DIR") or None,
            gipsy_max_workers=int(os.environ.get("GOTL_GIPSY_MAX_WORKERS", "4")),
            loaddef_root=os.environ.get("GOTL_LOADDEF_ROOT") or None,
            loaddef_data_dir=os.environ.get("GOTL_LOADDEF_DATA_DIR") or None,
        )
        db_path = Path(results_dir) / ".web" / "jobs.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            gotl=gotl_cfg,
            port=int(os.environ.get("GOTL_WEB_PORT", "8050")),
            max_concurrent_jobs=int(os.environ.get("GOTL_WEB_MAX_JOBS", "2")),
            db_path=db_path,
        )

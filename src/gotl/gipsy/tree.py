"""kinematicPPP.tree template management.

Bundles the GipsyX processing configuration template and copies it into
the working directory for gd2e.py.  The template has OCEANLOAD == On so
GipsyX applies OTL corrections during PPP for better position estimates.
The exact correction is backed out in post-processing using the same OTL
parameters (via hardisp) to recover the full tidal signal for VTide.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "kinematicPPP.tree"


def get_tree_template_path() -> Path:
    """Return the path to the bundled kinematicPPP.tree template."""
    if not _TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"kinematicPPP.tree template not found at {_TEMPLATE_PATH}"
        )
    return _TEMPLATE_PATH


def prepare_tree_dir(
    work_dir: Path,
    vmf1_dir: Path | None = None,
    rate_sec: int | None = None,
) -> Path:
    """Set up the Trees/ directory for gd2e.py.

    Copies the bundled kinematicPPP.tree to ``{work_dir}/Trees/ppp0_0.tree``.
    Optionally updates the VMF1dataDir path and the GLOBAL_DATA_RATE.

    Args:
        work_dir: Working directory for the processing run.
        vmf1_dir: Path to VMF1 troposphere data directory.
                  If provided, replaces the hardcoded cluster path.
        rate_sec: Override GLOBAL_DATA_RATE in the tree (seconds).
                  Should match the data record sampling rate.

    Returns:
        Path to the Trees/ directory.
    """
    trees_dir = Path(work_dir) / "Trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    dest = trees_dir / "ppp0_0.tree"

    content = _TEMPLATE_PATH.read_text()

    # If VMF1 data is available, switch from GMF to VMF1 mapping
    if vmf1_dir is not None:
        content = content.replace(
            "# VMF1dataDir is set at runtime if cfg.vmf1_dir is provided",
            f"VMF1dataDir {vmf1_dir}",
        )
        content = content.replace("Mapping GMF", "Mapping VMF1")

    # Override GLOBAL_DATA_RATE if provided
    if rate_sec is not None:
        content = content.replace(
            "GLOBAL_DATA_RATE == 300",
            f"GLOBAL_DATA_RATE == {rate_sec}",
        )

    dest.write_text(content)
    log.debug("Tree template written to %s", dest)
    return trees_dir

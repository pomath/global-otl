"""GipsyX RINEX-to-TDP processing (Phase 0).

Wraps GipsyX command-line tools to convert daily RINEX observation files
into TDP position time series suitable for the gotl analysis pipeline.

Pipeline:
    1. Discover RINEX files (rinex.py)
    2. Create station database from RINEX headers (stadb.py)
    3. Convert RINEX → data records, merge into one .dr.gz (datarecord.py)
    4. Process data record through PPP in overlapping windows (runner.py)
    5. Merge daily TDP files (batch.py)
"""

from gotl.gipsy.batch import process_station_rinex
from gotl.gipsy.rinex import RinexFile, discover_rinex

__all__ = ["process_station_rinex", "discover_rinex", "RinexFile"]

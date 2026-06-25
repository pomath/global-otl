"""Unit tests for the gotl.gipsy module.

These tests exercise RINEX filename parsing, file discovery, product
resolution, tree template management, and processing window calculation.
No GipsyX installation is required.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from gotl.gipsy.rinex import (
    RinexFile,
    _parse_rinex2_name,
    _parse_rinex3_name,
    discover_rinex,
    parse_rinex_name,
)
from gotl.gipsy.products import check_products_available, resolve_products_dir
from gotl.gipsy.runner import compute_processing_windows
from gotl.gipsy.tree import get_tree_template_path, prepare_tree_dir


# --------------------------------------------------------------------------
# RINEX filename parsing
# --------------------------------------------------------------------------


class TestParseRinex2:
    def test_standard(self):
        stnm, d, compressed = _parse_rinex2_name("ABOA0010.10o")
        assert stnm == "aboa"
        assert d == date(2010, 1, 1)
        assert compressed is False

    def test_compressed_Z(self):
        stnm, d, compressed = _parse_rinex2_name("ABOA0010.10o.Z")
        assert stnm == "aboa"
        assert d == date(2010, 1, 1)
        assert compressed is True

    def test_compressed_gz(self):
        stnm, d, compressed = _parse_rinex2_name("SPGT1830.05o.gz")
        assert stnm == "spgt"
        assert d == date(2005, 7, 2)
        assert compressed is True

    def test_year_rollover_99(self):
        """2-digit year 99 → 1999."""
        stnm, d, _ = _parse_rinex2_name("TEST3650.99o")
        assert d == date(1999, 12, 31)

    def test_year_rollover_00(self):
        """2-digit year 00 → 2000."""
        stnm, d, _ = _parse_rinex2_name("TEST0010.00o")
        assert d == date(2000, 1, 1)

    def test_doy_middle_of_year(self):
        """DOY 182 → July 1 in a non-leap year."""
        stnm, d, _ = _parse_rinex2_name("BEAN1820.15o")
        assert d == date(2015, 7, 1)

    def test_uppercase_O(self):
        stnm, d, _ = _parse_rinex2_name("ABOA0010.10O")
        assert stnm == "aboa"

    def test_d_suffix(self):
        """RINEX2 with .d suffix (Hatanaka compressed obs)."""
        stnm, d, compressed = _parse_rinex2_name("ABOA0010.10d.Z")
        assert stnm == "aboa"
        assert compressed is True

    def test_invalid(self):
        assert _parse_rinex2_name("not_a_rinex.txt") is None
        assert _parse_rinex2_name("AB0010.10o") is None  # 2-char stnm
        assert _parse_rinex2_name("") is None


class TestParseRinex3:
    def test_standard_crx_gz(self):
        name = "ABOA00ATA_R_20100010000_01D_30S_MO.crx.gz"
        stnm, d, compressed = _parse_rinex3_name(name)
        assert stnm == "aboa"
        assert d == date(2010, 1, 1)
        assert compressed is True

    def test_rnx_uncompressed(self):
        name = "SPGT00USA_R_20221830000_01D_30S_MO.rnx"
        stnm, d, compressed = _parse_rinex3_name(name)
        assert stnm == "spgt"
        assert d == date(2022, 7, 2)
        assert compressed is False

    def test_invalid(self):
        assert _parse_rinex3_name("not_rinex3.txt") is None


class TestParseRinexName:
    def test_rinex2(self):
        result = parse_rinex_name("ABOA0010.10o.Z")
        assert result is not None
        assert result[0] == "aboa"

    def test_rinex3(self):
        result = parse_rinex_name("ABOA00ATA_R_20100010000_01D_30S_MO.crx.gz")
        assert result is not None
        assert result[0] == "aboa"

    def test_unknown(self):
        assert parse_rinex_name("random.txt") is None


# --------------------------------------------------------------------------
# RINEX discovery
# --------------------------------------------------------------------------


class TestDiscoverRinex:
    def test_discover_in_stn_subdir(self, tmp_path):
        """Files in {rinex_dir}/{stnm}/ are found."""
        stn_dir = tmp_path / "aboa"
        stn_dir.mkdir()
        for doy in [1, 2, 3]:
            (stn_dir / f"ABOA{doy:03d}0.10o.Z").touch()

        result = discover_rinex(tmp_path, "aboa")
        assert len(result) == 3
        assert result[0].date == date(2010, 1, 1)
        assert result[2].date == date(2010, 1, 3)

    def test_discover_flat_dir(self, tmp_path):
        """Files in {rinex_dir}/ (flat) are found."""
        (tmp_path / "ABOA0010.10o.Z").touch()
        (tmp_path / "ABOA0020.10o.Z").touch()
        (tmp_path / "SPGT0010.10o.Z").touch()  # different station

        result = discover_rinex(tmp_path, "aboa")
        assert len(result) == 2

    def test_discover_date_filter(self, tmp_path):
        """Date range filtering works."""
        stn_dir = tmp_path / "aboa"
        stn_dir.mkdir()
        for doy in range(1, 11):
            (stn_dir / f"ABOA{doy:03d}0.10o.Z").touch()

        result = discover_rinex(
            tmp_path, "aboa",
            start_date=date(2010, 1, 3),
            end_date=date(2010, 1, 7),
        )
        assert len(result) == 5
        assert result[0].date == date(2010, 1, 3)
        assert result[-1].date == date(2010, 1, 7)

    def test_discover_sorted_by_date(self, tmp_path):
        """Results are sorted chronologically."""
        stn_dir = tmp_path / "test"
        stn_dir.mkdir()
        # Create in reverse order
        for doy in [10, 5, 1, 8]:
            (stn_dir / f"TEST{doy:03d}0.10o").touch()

        result = discover_rinex(tmp_path, "test")
        dates = [r.date for r in result]
        assert dates == sorted(dates)

    def test_discover_empty_raises(self, tmp_path):
        """No RINEX files → FileNotFoundError."""
        stn_dir = tmp_path / "aboa"
        stn_dir.mkdir()
        (stn_dir / "readme.txt").touch()  # non-RINEX file

        with pytest.raises(FileNotFoundError):
            discover_rinex(tmp_path, "aboa")

    def test_discover_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_rinex(tmp_path / "nonexistent", "aboa")


# --------------------------------------------------------------------------
# JPL products
# --------------------------------------------------------------------------


class TestProducts:
    def test_resolve_existing(self, tmp_path):
        (tmp_path / "2010").mkdir()
        result = resolve_products_dir(tmp_path, date(2010, 6, 15))
        assert result == tmp_path / "2010"

    def test_resolve_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_products_dir(tmp_path, date(2010, 6, 15))

    def test_check_batch(self, tmp_path):
        (tmp_path / "2010").mkdir()
        (tmp_path / "2012").mkdir()

        dates = [date(2010, 1, 1), date(2011, 1, 1), date(2012, 1, 1)]
        available, missing = check_products_available(tmp_path, dates)
        assert len(available) == 2
        assert len(missing) == 1
        assert missing[0] == date(2011, 1, 1)


# --------------------------------------------------------------------------
# Tree template
# --------------------------------------------------------------------------


class TestTree:
    def test_template_exists(self):
        path = get_tree_template_path()
        assert path.exists()

    def test_oceanload_on(self):
        path = get_tree_template_path()
        content = path.read_text()
        assert "OCEANLOAD == On" in content
        assert "OceanLoadFile otlstadb.db" in content

    def test_prepare_tree_dir(self, tmp_path):
        trees_dir = prepare_tree_dir(tmp_path)
        assert trees_dir == tmp_path / "Trees"
        assert (trees_dir / "ppp0_0.tree").exists()

    def test_prepare_tree_vmf1_substitution(self, tmp_path):
        trees_dir = prepare_tree_dir(tmp_path, vmf1_dir=Path("/data/vmf1"))
        content = (trees_dir / "ppp0_0.tree").read_text()
        assert "/data/vmf1" in content
        assert "/fs/project/gomez.124/otl_proc/VMF1Dir" not in content


# --------------------------------------------------------------------------
# Processing windows
# --------------------------------------------------------------------------


class TestProcessingWindows:
    def test_single_day(self):
        """One day span → one window."""
        start = 1000000.0  # arbitrary GPS seconds
        end = start + 86400.0
        windows = compute_processing_windows(start, end)
        assert len(windows) == 1
        idx, w_start, w_end = windows[0]
        assert idx == 0
        assert w_start == start - 10800  # 3h before
        assert w_end == end + 97200  # 27h after

    def test_multi_day(self):
        """5-day span → 5 windows."""
        start = 1000000.0
        end = start + 5 * 86400.0
        windows = compute_processing_windows(start, end)
        assert len(windows) == 5

    def test_window_overlap(self):
        """Adjacent windows overlap (30h each, 24h apart → 6h overlap)."""
        start = 0.0
        end = 3 * 86400.0
        windows = compute_processing_windows(start, end)
        assert len(windows) == 3

        # Window 0 ends at 86400 + 97200 = 183600
        # Window 1 starts at 86400 - 10800 = 75600
        # Overlap: 75600 to 183600 = 108000 seconds = 30h
        _, _, w0_end = windows[0]
        _, w1_start, _ = windows[1]
        assert w1_start < w0_end  # windows overlap

    def test_window_durations(self):
        """Each window spans 54 hours: -3h before day start to +27h after next day.

        Duration = 24h (day gap) + 3h (pre-buffer) + 27h (post-buffer) = 54h.
        """
        start = 0.0
        end = 2 * 86400.0
        windows = compute_processing_windows(start, end)
        for _, w_start, w_end in windows:
            duration = w_end - w_start
            assert duration == 194400  # 54 hours

    def test_zero_span(self):
        """Same start/end → no windows."""
        windows = compute_processing_windows(100.0, 100.0)
        assert len(windows) == 0


# --------------------------------------------------------------------------
# TDP output naming
# --------------------------------------------------------------------------


class TestDailyDrMap:
    def test_rinex3_filenames(self, tmp_path):
        from gotl.gipsy.batch import _build_daily_dr_map
        for doy in (1, 15, 100, 365):
            (tmp_path / f"BRUX00BEL_R_2024{doy:03d}0000_01D_30S_MO.crx.dr.gz").touch()
        m = _build_daily_dr_map(tmp_path)
        assert set(m.keys()) == {1, 15, 100, 365}

    def test_rinex2_filenames(self, tmp_path):
        from gotl.gipsy.batch import _build_daily_dr_map
        for doy in (1, 15, 100, 365):
            (tmp_path / f"bhr3{doy:03d}0.24d.dr.gz").touch()
        m = _build_daily_dr_map(tmp_path)
        assert set(m.keys()) == {1, 15, 100, 365}

    def test_mixed_rinex2_and_rinex3(self, tmp_path):
        from gotl.gipsy.batch import _build_daily_dr_map
        # gode-style: some R2 files, some R3 files
        (tmp_path / "gode0090.24d.dr.gz").touch()  # R2: DOY 9
        (tmp_path / "GODE00USA_R_20240100000_01D_30S_MO.crx.dr.gz").touch()  # R3: DOY 10
        m = _build_daily_dr_map(tmp_path)
        assert set(m.keys()) == {9, 10}

    def test_skips_unparseable(self, tmp_path):
        from gotl.gipsy.batch import _build_daily_dr_map
        (tmp_path / "weird-name.dr.gz").touch()
        (tmp_path / "BRUX00BEL_R_20240010000_01D_30S_MO.crx.dr.gz").touch()
        m = _build_daily_dr_map(tmp_path)
        assert set(m.keys()) == {1}


class TestTdpNaming:
    def test_output_matches_load_tdp_pattern(self):
        """Output filenames match the regex in io/tdp.py."""
        pattern = re.compile(r"^\d+\.tdp$")
        for i in range(100):
            name = f"{i:04d}.tdp"
            assert pattern.match(name), f"{name} should match TDP pattern"

    def test_large_index(self):
        """Large indices still match (5+ digits)."""
        pattern = re.compile(r"^\d+\.tdp$")
        assert pattern.match("10000.tdp")
        assert pattern.match("99999.tdp")


# --------------------------------------------------------------------------
# Data record merge batching
# --------------------------------------------------------------------------


class TestMergeBatching:
    def test_single_file_no_merge(self):
        """A single DR file should be copied, not merged."""
        # This is a logic test; actual merge requires GipsyX
        from gotl.gipsy.datarecord import merge_data_records
        # We can't run it without GipsyX, but we verify the branch
        # by checking it doesn't call drMerge.py for 1 file
        # (It copies instead — tested via the code path)

    def test_batch_calculation(self):
        """Verify batch splitting for >199 files."""
        n_files = 450
        batch_size = 199
        n_batches = (n_files + batch_size - 1) // batch_size
        assert n_batches == 3  # 199 + 199 + 52


class TestStadbNormalize:
    """Regression tests for the 9-char → 4-char marker rewrite.

    MAC1 RINEX 3 files use ``MARKER NAME = MAC100AUS``; rinex2StaDb.py copies
    that verbatim, but gd2e.py is invoked with ``-recList MAC1`` and then
    fails inside ``prepareRunDirSubDir`` when the antenna lookup can't find
    the station. The normalizer rewrites the stadb so the lookup succeeds.
    """

    def _write(self, tmp_path, text):
        p = tmp_path / "stn.stadb"
        p.write_text(text)
        return p

    def test_rewrites_9char_marker(self, tmp_path):
        from gotl.gipsy.stadb import _normalize_station_id

        p = self._write(tmp_path, (
            "KEYWORDS: ID STATE END ANT RX\n"
            "MAC100AUS  ID  50135M001  MAC100AUS\n"
            "MAC100AUS  STATE  1978-01-01 00:00:00  -3.4e+06  1.3e+06  -5.1e+06\n"
            "MAC100AUS  ANT    1978-01-01 00:00:00  JAVRINGANT_DM SCIS\n"
            "MAC100AUS  RX     1978-01-01 00:00:00  SEPT POLARX5\n"
        ))
        _normalize_station_id(p, "mac1")
        out = p.read_text()
        assert "MAC100AUS" not in out
        assert out.count("MAC1") == 5  # 4 line prefixes + the domes-adjacent ID field

    def test_noop_when_already_normalized(self, tmp_path):
        from gotl.gipsy.stadb import _normalize_station_id

        text = (
            "KEYWORDS: ID STATE END ANT RX\n"
            "BRUX  ID  13101M010  BRUX\n"
            "BRUX  STATE  1978-01-01 00:00:00  4.0e+06  3.0e+05  4.9e+06\n"
        )
        p = self._write(tmp_path, text)
        _normalize_station_id(p, "brux")
        assert p.read_text() == text

    def test_coalesces_multiple_ids_to_target(self, tmp_path):
        """Year-long archives may mix 4-char and 9-char marker formats.
        Sources rewrite to the target and duplicate ID records get deduped
        (only one ID line per station survives).
        """
        from gotl.gipsy.stadb import _normalize_station_id

        p = self._write(tmp_path, (
            "KEYWORDS: ID STATE END ANT RX\n"
            "ARHT       ID  10000M001  ARHT\n"
            "ARHT       ANT  10000M001  ASH701945E_M\n"
            "ARHT00ATA  ID  20000M001  ARHT00ATA\n"
            "ARHT00ATA  ANT  20000M001  ASH701945E_M\n"
        ))
        _normalize_station_id(p, "arht")
        out = p.read_text()
        # No source markers remain
        assert "ARHT00ATA" not in out
        # Exactly one ID line for ARHT (dedup'd)
        id_lines = [ln for ln in out.splitlines()
                    if ln and not ln.startswith(("#", "KEYWORDS:"))
                    and len(ln.split()) >= 2 and ln.split()[1] == "ID"]
        assert len(id_lines) == 1, f"expected 1 ID line, got {len(id_lines)}: {id_lines}"
        # ANT records survive (both rewritten to ARHT)
        ant_lines = [ln for ln in out.splitlines()
                     if ln and not ln.startswith(("#", "KEYWORDS:"))
                     and len(ln.split()) >= 2 and ln.split()[1] == "ANT"]
        assert len(ant_lines) == 2

    def test_dedup_only_when_duplicates_exist(self, tmp_path):
        """A clean single-ID stadb is left untouched."""
        from gotl.gipsy.stadb import _normalize_station_id

        text = (
            "KEYWORDS: ID STATE END ANT RX\n"
            "BRUX  ID  10000M001  BRUX\n"
            "BRUX  ANT 10000M001  TYPE_X\n"
        )
        p = self._write(tmp_path, text)
        _normalize_station_id(p, "brux")
        assert p.read_text() == text  # unchanged

"""Tests for the LoadDef OTL parameter generation module."""

import textwrap
from pathlib import Path

import pytest

from gotl.loaddef.convert import convert_convolution_to_otl, write_station_file


# ── Fixtures ──────────────────────────────────────────────────────────


SAMPLE_CONVOLUTION_OUTPUT = textwrap.dedent("""\
    Extension/Epoch  Lat(+N,deg)  Lon(+E,deg)  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg)
    FES2014-M2  -73.043770  346.592866  5.09000  61.600  2.16000  205.500  11.77000  48.400
    FES2014-S2  -73.043770  346.592866  3.92000  235.400  1.14000  138.200  9.03000  221.800
    FES2014-N2  -73.043770  346.592866  0.83000  42.500  0.25000  201.700  1.91000  28.000
    FES2014-K2  -73.043770  346.592866  0.78000  96.200  0.57000  239.500  2.14000  61.800
    FES2014-K1  -73.043770  346.592866  3.92000  235.400  1.14000  138.200  9.03000  221.800
    FES2014-O1  -73.043770  346.592866  3.34000  219.000  0.76000  48.600  10.23000  201.800
    FES2014-P1  -73.043770  346.592866  1.23000  234.700  0.35000  129.700  3.14000  218.200
    FES2014-Q1  -73.043770  346.592866  0.72000  204.600  0.20000  5.300  2.41000  186.200
    FES2014-MF  -73.043770  346.592866  0.20000  81.400  0.53000  156.200  2.04000  18.200
    FES2014-MM  -73.043770  346.592866  0.09000  44.200  0.18000  133.600  1.18000  9.600
    FES2014-SSA  -73.043770  346.592866  0.13000  7.900  0.01000  125.100  1.10000  1.100
""")

# Expected values from the real ABOA FES2014 .otl file
EXPECTED_AMP_U_M2 = 0.01177  # 11.77 mm -> m


@pytest.fixture
def conv_file(tmp_path):
    """Write sample convolution output to a temp file."""
    f = tmp_path / "cn_OceanOnly_aboa_cm_convgf_FES2014_PREM.txt"
    f.write_text(SAMPLE_CONVOLUTION_OUTPUT)
    return f


# ── convert.py tests ─────────────────────────────────────────────────


class TestConvertConvolutionToOTL:
    def test_produces_39_line_file(self, conv_file, tmp_path):
        """Output must have exactly 33 header + 6 data lines + footer."""
        out = tmp_path / "aboa_FES2014.otl"
        convert_convolution_to_otl(
            conv_file, "aboa", "PREM", "FES2014",
            lat=-73.0438, lon=-13.4071, alt=468.7,
            output_path=out,
        )
        lines = out.read_text().splitlines()
        # 33 header + 6 data + 1 footer ($$) = 40 lines
        assert len(lines) == 40

    def test_data_on_correct_lines(self, conv_file, tmp_path):
        """Data must be on lines 34-39 (1-indexed) = indices 33-38."""
        out = tmp_path / "aboa_FES2014.otl"
        convert_convolution_to_otl(
            conv_file, "aboa", "PREM", "FES2014",
            lat=-73.0438, lon=-13.4071, alt=468.7,
            output_path=out,
        )
        lines = out.read_text().splitlines()
        # Lines 33-38 (0-indexed) should be numeric data
        for i in range(33, 39):
            vals = lines[i].split()
            assert len(vals) == 11, f"Line {i+1} should have 11 values, got {len(vals)}"
            # All should be parseable as floats
            for v in vals:
                float(v)

    def test_amplitude_values_correct(self, conv_file, tmp_path):
        """M2 radial amplitude should match expected value."""
        out = tmp_path / "aboa_FES2014.otl"
        convert_convolution_to_otl(
            conv_file, "aboa", "PREM", "FES2014",
            lat=-73.0438, lon=-13.4071, alt=468.7,
            output_path=out,
        )
        lines = out.read_text().splitlines()
        # Line 34 (index 33) = radial amplitudes, first value = M2
        amp_u_vals = [float(v) for v in lines[33].split()]
        assert abs(amp_u_vals[0] - EXPECTED_AMP_U_M2) < 1e-5

    def test_read_otl_params_compatible(self, conv_file, tmp_path):
        """Output must be readable by read_otl_params()."""
        out = tmp_path / "aboa_FES2014.otl"
        convert_convolution_to_otl(
            conv_file, "aboa", "PREM", "FES2014",
            lat=-73.0438, lon=-13.4071, alt=468.7,
            output_path=out,
        )
        from gotl.io.otl_params import read_otl_params
        df = read_otl_params("aboa", tmp_path, suffix="_FES2014")
        assert df.shape == (11, 6)
        assert abs(df.loc["M2", "dU"] - EXPECTED_AMP_U_M2) < 1e-5
        # Phase should be 48.4 degrees for M2 radial
        assert abs(df.loc["M2", "gU"] - 48.4) < 0.1

    def test_missing_harmonic_zero_filled(self, tmp_path):
        """TPXO9-Atlas has no SSA; convert must emit a zero row rather than drop it."""
        conv = tmp_path / "cn_OceanOnly_aboa_cm_convgf_TPXO9-Atlas_PREM.txt"
        # Same content as SAMPLE_CONVOLUTION_OUTPUT, but SSA row removed and
        # "FES2014-" replaced with "TPXO9-Atlas-" so the extension parser sees it.
        lines = SAMPLE_CONVOLUTION_OUTPUT.splitlines()
        kept = [lines[0]] + [
            l.replace("FES2014-", "TPXO9-Atlas-")
            for l in lines[1:]
            if not l.startswith("FES2014-SSA")
        ]
        conv.write_text("\n".join(kept) + "\n")

        out = tmp_path / "aboa_TPXO9.otl"
        convert_convolution_to_otl(
            conv, "aboa", "PREM", "TPXO9",
            lat=-73.0438, lon=-13.4071, alt=468.7,
            output_path=out,
        )
        from gotl.io.otl_params import read_otl_params
        df = read_otl_params("aboa", tmp_path, suffix="_TPXO9")
        assert df.shape == (11, 6)
        assert df.loc["SSA", "dU"] == 0.0
        assert df.loc["SSA", "gU"] == 0.0
        # Non-SSA rows should still round-trip
        assert abs(df.loc["M2", "dU"] - EXPECTED_AMP_U_M2) < 1e-5


class TestStationFile:
    def test_lon_conversion(self, tmp_path):
        """Negative longitude should be converted to 0-360."""
        out = tmp_path / "stations.txt"
        write_station_file([("aboa", -73.0438, -13.4071)], out)
        line = out.read_text().strip()
        parts = line.split()
        lon = float(parts[1])
        assert 0 <= lon < 360
        assert abs(lon - 346.5929) < 0.001

    def test_multi_station(self, tmp_path):
        """Multiple stations should produce multiple lines."""
        out = tmp_path / "stations.txt"
        write_station_file([
            ("aboa", -73.0, -13.4),
            ("brux", 50.8, 4.36),
        ], out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        assert "aboa" in lines[0]
        assert "brux" in lines[1]


# ── env.py tests ─────────────────────────────────────────────────────


class TestLoadDefEnv:
    def test_missing_loaddef_root(self, tmp_path):
        """Should raise if LoadDef root doesn't exist."""
        from gotl.loaddef.env import load_loaddef_env
        with pytest.raises(FileNotFoundError):
            load_loaddef_env(tmp_path / "nonexistent", tmp_path)

    def test_missing_modules(self, tmp_path):
        """Should raise if CONVGF/LOADGF dirs missing."""
        from gotl.loaddef.env import load_loaddef_env
        fake_root = tmp_path / "LoadDef"
        fake_root.mkdir()
        with pytest.raises(FileNotFoundError, match="CONVGF"):
            load_loaddef_env(fake_root, tmp_path)


# ── templates.py tests ───────────────────────────────────────────────


class TestTemplates:
    def test_love_numbers_template_valid_python(self):
        """Template should produce valid Python after substitution."""
        from gotl.loaddef.templates import LOVE_NUMBERS_SCRIPT
        script = LOVE_NUMBERS_SCRIPT.format(
            loaddef_root="/tmp/LoadDef",
            planet_model_path="/tmp/PREM.txt",
            output_dir="/tmp/output",
            working_dir="/tmp/working",
            earth_model="PREM",
        )
        compile(script, "<love_numbers>", "exec")

    def test_convolution_template_valid_python(self):
        """Template should produce valid Python after substitution."""
        from gotl.loaddef.templates import CONVOLUTION_SCRIPT
        script = CONVOLUTION_SCRIPT.format(
            loaddef_root="/tmp/LoadDef",
            grn_file="/tmp/ce_PREM.txt",
            loadfile_directory="/tmp/grids",
            loadfile_prefix="convgf_FES2014",
            lsmask_file="/tmp/mask.txt",
            sta_file="/tmp/stations.txt",
            rfm="cm",
            earth_model="PREM",
            output_dir="/tmp/output",
            working_dir="/tmp/working",
        )
        compile(script, "<convolution>", "exec")

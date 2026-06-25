"""Convert LoadDef convolution output to Harpos .otl format.

The Harpos/Chalmers format has 33 header lines + 6 data lines.
Lines 34-39 (1-indexed) contain the tidal loading coefficients
that ``read_otl_params()`` and ``hardisp`` consume.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

# 11 tidal constituents in Harpos order
HARMONICS = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]


def convert_convolution_to_otl(
    convolution_file: Path,
    station: str,
    earth_model: str,
    tide_model: str,
    lat: float,
    lon: float,
    alt: float,
    output_path: Path,
) -> Path:
    """Convert LoadDef convolution output to Harpos .otl format.

    Args:
        convolution_file: LoadDef convolution text output.
        station: 4-letter station code.
        earth_model: Earth model name (e.g. "PREM").
        tide_model: Tide model name (e.g. "FES2014").
        lat: Station latitude (degrees).
        lon: Station longitude (degrees, -180 to 180).
        alt: Station altitude (metres).
        output_path: Where to write the .otl file.

    Returns:
        Path to the created .otl file.
    """
    # Parse convolution output
    data = Path(convolution_file).read_text().splitlines()

    # Build lookup: harmonic name -> row values
    # Format: Extension/Epoch  Lat  Lon  E-Amp(mm)  E-Pha(deg)  N-Amp(mm)  N-Pha(deg)  V-Amp(mm)  V-Pha(deg)
    harmonic_data: dict[str, list[str]] = {}
    for line in data:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("Extension"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        # The harmonic name is after the last hyphen in the first column
        ext = parts[0]
        # e.g. "FES2014-M2" -> "M2", "TPXO9-Atlas-M2" -> "M2"
        harmonic = ext.rsplit("-", 1)[-1].upper()
        harmonic_data[harmonic] = parts

    # Extract amplitudes and phases for each harmonic in order
    amp_u, amp_w, amp_s = [], [], []
    pha_u, pha_w, pha_s = [], [], []

    for h in HARMONICS:
        row = harmonic_data.get(h)
        if row is None:
            # Missing harmonic: zero values
            amp_u.append(0.0)
            amp_w.append(0.0)
            amp_s.append(0.0)
            pha_u.append(0.0)
            pha_w.append(0.0)
            pha_s.append(0.0)
        else:
            # Columns: 0=ext 1=lat 2=lon 3=E-Amp 4=E-Pha 5=N-Amp 6=N-Pha 7=V-Amp 8=V-Pha
            # V = radial (U), E = tangential EW (W), N = tangential NS (S)
            amp_u.append(float(row[7]) / 1e3)  # mm -> m
            amp_w.append(float(row[3]) / 1e3)
            amp_s.append(float(row[5]) / 1e3)
            pha_u.append(float(row[8]))
            pha_w.append(float(row[4]))
            pha_s.append(float(row[6]))

    # Format amplitudes: leading dot, 5 decimal places (e.g. ".01005")
    def fmt_amp(vals: list[float]) -> str:
        parts = []
        for v in vals:
            s = f"{abs(v):.5f}"
            # Strip leading zero: "0.01005" -> ".01005"
            if s.startswith("0"):
                s = s[1:]
            parts.append(s)
        return "  " + " ".join(parts)

    # Format phases: 6 chars wide, 1 decimal place
    def fmt_pha(vals: list[float]) -> str:
        return "  " + " ".join(f"{v:6.1f}" for v in vals)

    # Build 33-line Chalmers/Harpos header
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "$$ Ocean loading displacement",
        "$$",
        f"$$ Computed by LoadDef ({earth_model} Greens function)",
        f"$$ Tide model: {tide_model}",
        "$$",
        "$$ COLUMN ORDER:  M2  S2  N2  K2  K1  O1  P1  Q1  MF  MM SSA",
        "$$",
        "$$ ROW ORDER:",
        "$$ AMPLITUDES (m)",
        "$$   RADIAL",
        "$$   TANGENTL    EW",
        "$$   TANGENTL    NS",
        "$$ PHASES (degrees)",
        "$$   RADIAL",
        "$$   TANGENTL    EW",
        "$$   TANGENTL    NS",
        "$$",
        "$$ Displacement is defined positive in upwards, South and West direction.",
        "$$ The phase lag is relative to Greenwich and lags positive.",
        f"$$ {earth_model} Greens function is used. Ocean-only loading (land masked).",
        "$$",
        "$$ Reference frame: centre of mass (CM)",
        "$$",
        "$$ CMC:  NO (corr.tide centre of mass)",
        "$$",
        f"$$ {tide_model}: m2 s2 n2 k2 k1 o1 p1 q1 Mf Mm Ssa",
        "$$",
        "$$ END HEADER",
        "$$",
        f"  {station.lower()}",
        f"$$ {tide_model} ID:{now}",
        f"$$ Computed using LoadDef (Martens et al.)",
        f"$$ {station.lower():20s} RADI TANG  lon/lat: {lon:8.4f}  {lat:8.4f}  {alt:9.3f}",
    ]

    assert len(header) == 33, f"Header must be 33 lines, got {len(header)}"

    # 6 data lines (lines 34-39)
    data_lines = [
        fmt_amp(amp_u),
        fmt_amp(amp_w),
        fmt_amp(amp_s),
        fmt_pha(pha_u),
        fmt_pha(pha_w),
        fmt_pha(pha_s),
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header + data_lines) + "\n$$\n")

    log.info("OTL file written: %s", output_path)
    return output_path


def write_station_file(
    stations: list[tuple[str, float, float]],
    output_path: Path,
) -> Path:
    """Write a LoadDef station location file.

    Args:
        stations: List of (station_code, lat, lon) tuples.
                  Longitude is -180 to 180; converted to 0-360 internally.
        output_path: Where to write the file.

    Returns:
        Path to the created file.
    """
    lines = []
    for stnm, lat, lon in stations:
        lon360 = lon % 360  # Convert to 0-360
        lines.append(f"{lat:.6f}  {lon360:.6f}  {stnm.lower()}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path

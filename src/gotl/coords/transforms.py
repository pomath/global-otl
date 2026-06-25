"""Coordinate system transforms: ECEF ↔ geodetic (LLA) and ENU.

Replaces: MATLAB Mapping Toolbox functions ecef2lla() and ecef2enu()
Library: pymap3d (WGS-84 ellipsoid, matching MATLAB wgs84Ellipsoid)
"""

import numpy as np
import pymap3d


def ecef_to_lla(x, y, z):
    """Convert ECEF coordinates to geodetic latitude, longitude, altitude.

    Matches MATLAB: lla = ecef2lla([x, y, z])
    Uses WGS-84 ellipsoid.

    Args:
        x, y, z: ECEF coordinates in metres (scalar or array)

    Returns:
        (lat_deg, lon_deg, alt_m): geodetic latitude (°N), longitude (°E), altitude (m)
    """
    lat, lon, alt = pymap3d.ecef2geodetic(x, y, z)
    return np.asarray(lat), np.asarray(lon), np.asarray(alt)


def ecef_to_enu(x, y, z, lat0_deg, lon0_deg, alt0_m):
    """Convert ECEF coordinates to local East-North-Up (ENU) relative to reference point.

    Matches MATLAB: [e, n, u] = ecef2enu(x, y, z, lat0, lon0, alt0, wgs84Ellipsoid)
    Uses WGS-84 ellipsoid.

    Args:
        x, y, z: ECEF coordinates in metres (scalar or array)
        lat0_deg: reference geodetic latitude in degrees
        lon0_deg: reference geodetic longitude in degrees
        alt0_m: reference altitude in metres

    Returns:
        (e, n, u): East, North, Up displacement in metres
    """
    e, n, u = pymap3d.ecef2enu(x, y, z, lat0_deg, lon0_deg, alt0_m)
    return np.asarray(e), np.asarray(n), np.asarray(u)

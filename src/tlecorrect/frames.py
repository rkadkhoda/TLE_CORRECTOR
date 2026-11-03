"""Coordinate frames: TEME <-> ECEF <-> ENU/az-el and the RTN basis.

Conventions were validated against the golden ground-truth CSV: using the
proper ECEF velocity (rotated TEME velocity minus the Earth-rotation term)
the predicted Doppler matches the generator truth to ~0.15 Hz RMS.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "WGS84_A_M",
    "WGS84_F",
    "OMEGA_EARTH",
    "C_LIGHT",
    "gmst_rad",
    "geodetic_to_ecef",
    "ecef_to_geodetic",
    "enu_basis",
    "teme_to_ecef_state",
    "teme_station_state",
    "range_and_rate",
    "rtn_basis",
    "az_el_range",
]

WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
OMEGA_EARTH = 7.29211514670698e-5  # rad/s (IERS nominal)
C_LIGHT = 299792458.0  # m/s


def gmst_rad(jd_ut1: float, fr: float = 0.0) -> float:
    """Greenwich Mean Sidereal Time (rad) from a split Julian date (UT1~UTC).

    ``jd_ut1 + fr`` must not be formed first: a full Julian date near 2.46e6
    has a float64 resolution of about 40 microseconds, which is a centimetre
    of station motion and, over a 10 ms sampling interval, a metre-per-second
    error in the numerical range derivative.  Using the reduced day count
    keeps the GMST continuous at the microsecond level.
    """
    d = (float(jd_ut1) - 2451545.0) + float(fr)
    tut1 = d / 36525.0
    temp = (-6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1
            + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 67310.54841)
    temp = (temp * (np.pi / 180.0) / 240.0) % (2.0 * np.pi)
    return float(temp)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """WGS84 geodetic -> ECEF position (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    a = WGS84_A_M
    e2 = WGS84_F * (2.0 - WGS84_F)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - e2) + alt_m) * np.sin(lat)
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """ECEF (metres) -> (lat_deg, lon_deg, alt_m) via Bowring's method."""
    x, y, z = r_ecef
    a = WGS84_A_M
    b = a * (1.0 - WGS84_F)
    e2 = WGS84_F * (2.0 - WGS84_F)
    ep2 = (a * a - b * b) / (b * b)
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    theta = np.arctan2(z * a, p * b)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + ep2 * b * st**3, p - e2 * a * ct**3)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def enu_basis(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rows are East, North, Up unit vectors in ECEF."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.array([-so, co, 0.0])
    north = np.array([-sl * co, -sl * so, cl])
    up = np.array([cl * co, cl * so, sl])
    return np.vstack([east, north, up])


def teme_to_ecef_state(r_teme_km: np.ndarray, v_teme_kms: np.ndarray, jd: float,
                       fr: float = 0.0):
    """TEME position/velocity (km, km/s) -> ECEF position (km) and velocity (km/s).

    Uses GMST for the rotation and includes the Earth-rotation term so the
    ECEF velocity is the physically correct one.
    """
    g = gmst_rad(jd, fr)
    cg, sg = np.cos(g), np.sin(g)
    rot = np.array([[cg, sg, 0.0], [-sg, cg, 0.0], [0.0, 0.0, 1.0]])
    r_ecef = rot @ np.asarray(r_teme_km, dtype=float)
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    v_ecef = rot @ np.asarray(v_teme_kms, dtype=float) - np.cross(w, r_ecef * 1000.0) / 1000.0
    return r_ecef, v_ecef


def teme_to_ecef_states(r_teme_km: np.ndarray, v_teme_kms: np.ndarray,
                        jd: np.ndarray, fr: np.ndarray | None = None):
    """Vectorised TEME -> ECEF state conversion.

    ``r_teme_km``/``v_teme_kms`` have shape (N, 3); ``jd`` has shape (N,).
    Returns (r_ecef_km, v_ecef_kms) with shape (N, 3).
    """
    r = np.atleast_2d(np.asarray(r_teme_km, dtype=float))
    v = np.atleast_2d(np.asarray(v_teme_kms, dtype=float))
    jd_arr = np.atleast_1d(np.asarray(jd, dtype=float))
    if fr is None:
        g = np.array([gmst_rad(float(j)) for j in jd_arr])
    else:
        fr_arr = np.atleast_1d(np.asarray(fr, dtype=float))
        g = np.array([gmst_rad(float(a), float(b))
                      for a, b in zip(jd_arr, fr_arr, strict=True)])
    cg, sg = np.cos(g), np.sin(g)
    r_ecef = np.empty_like(r)
    r_ecef[:, 0] = cg * r[:, 0] + sg * r[:, 1]
    r_ecef[:, 1] = -sg * r[:, 0] + cg * r[:, 1]
    r_ecef[:, 2] = r[:, 2]
    v_ecef = np.empty_like(v)
    v_ecef[:, 0] = cg * v[:, 0] + sg * v[:, 1]
    v_ecef[:, 1] = -sg * v[:, 0] + cg * v[:, 1]
    v_ecef[:, 2] = v[:, 2]
    # ECEF velocity = rotated TEME velocity - omega x r_ecef
    w_cross = np.empty_like(r_ecef)
    w_cross[:, 0] = -OMEGA_EARTH * r_ecef[:, 1]
    w_cross[:, 1] = OMEGA_EARTH * r_ecef[:, 0]
    w_cross[:, 2] = 0.0
    v_ecef -= w_cross
    return r_ecef, v_ecef


def teme_station_state(lat_deg: float, lon_deg: float, alt_m: float, jd: float,
                       fr: float = 0.0):
    """Station position/velocity expressed in TEME (km, km/s)."""
    r_ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m) / 1000.0
    g = gmst_rad(jd, fr)
    cg, sg = np.cos(g), np.sin(g)
    inv = np.array([[cg, -sg, 0.0], [sg, cg, 0.0], [0.0, 0.0, 1.0]])
    r_teme = inv @ r_ecef
    w = np.array([0.0, 0.0, OMEGA_EARTH])
    v_teme = np.cross(w, r_teme)
    return r_teme, v_teme


def range_and_rate(r_sat_km: np.ndarray, v_sat_kms: np.ndarray, sta_ecef_m: np.ndarray,
                   lat_deg: float, lon_deg: float, jd: float) -> tuple[float, float]:
    """Slant range (m) and range-rate (m/s, positive when receding) for an ECEF sat state.

    ``v_sat_kms`` must be the *proper* ECEF velocity (km/s). The station is static
    in ECEF, so the range rate is simply the ECEF velocity projected on the LOS.
    """
    los_m = r_sat_km * 1000.0 - sta_ecef_m
    rng = float(np.linalg.norm(los_m))
    u = los_m / rng
    rr = float(np.dot(v_sat_kms * 1000.0, u))
    return rng, rr


def rtn_basis(r_km: np.ndarray, v_kms: np.ndarray) -> np.ndarray:
    """Return (R, T, N) unit vectors as rows. R radial/up, N cross-track, T=NxR."""
    r = np.asarray(r_km, dtype=float)
    v = np.asarray(v_kms, dtype=float)
    r_hat = r / np.linalg.norm(r)
    h = np.cross(r, v)
    n_hat = h / np.linalg.norm(h)
    t_hat = np.cross(n_hat, r_hat)
    return np.vstack([r_hat, t_hat, n_hat])


def az_el_range(r_sat_km: np.ndarray, sta_ecef_m: np.ndarray, lat_deg: float, lon_deg: float):
    """Topocentric azimuth/ elevation (deg) and range (m)."""
    los = r_sat_km * 1000.0 - sta_ecef_m
    enu = enu_basis(lat_deg, lon_deg) @ los
    east, north, up = enu
    rng = float(np.linalg.norm(enu))
    el = float(np.degrees(np.arcsin(up / rng)))
    az = float(np.degrees(np.arctan2(east, north)) % 360.0)
    return az, el, rng

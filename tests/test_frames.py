import csv

import numpy as np
import pytest

from tlecorrect.frames import (
    az_el_range,
    ecef_to_geodetic,
    geodetic_to_ecef,
    rtn_basis,
    teme_to_ecef_state,
)
from tlecorrect.sgp4wrap import OrbitModel
from tlecorrect.timeutil import jd_split_many, parse_iso
from tlecorrect.tle import parse_tle

L1 = "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999"
L2 = "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912"


def test_geodetic_ecef_round_trip():
    lat, lon, alt = 35.7, 51.4, 1200.0
    r = geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = ecef_to_geodetic(r)
    assert abs(lat2 - lat) < 1e-9
    assert abs(lon2 - lon) < 1e-9
    assert abs(alt2 - alt) < 1e-6


def test_rtn_orthonormal():
    r = np.array([6900.0, 353.0, 2080.0])
    v = np.array([-2.06, -1.31, 7.02])
    basis = rtn_basis(r, v)
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-12)
    assert np.allclose(basis[0], r / np.linalg.norm(r))


def _load_truth(path):
    t, d, rng, el = [], [], [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(parse_iso(row["iso_time"]))
            d.append(float(row["doppler_hz"]))
            rng.append(float(row["range_m"]))
            el.append(float(row["el_deg"]))
    return t, np.array(d), np.array(rng), np.array(el)


def test_predicted_doppler_matches_golden_truth(golden_out):
    """The forward physics must reproduce the generator truth to sub-Hz RMS."""
    truth = golden_out / "NOAA19_tehran_20261021T181639_truth.csv"
    if not truth.exists():
        pytest.skip("golden truth csv missing")
    tle = parse_tle(open(golden_out / "true.tle").read())
    model = OrbitModel(tle)
    times, dop, rng_true, el_true = _load_truth(truth)
    jd_all, fr_all = jd_split_many(times)
    sta = type("S", (), {"lat_deg": 35.7, "lon_deg": 51.4, "alt_m": 1200.0})()
    rng, rr = model.range_rate(model.base_elements, sta, jd_all, fr_all)
    fd = -137.1e6 * rr / 299792458.0
    err = fd - dop
    assert np.sqrt(np.mean(err**2)) < 1.0
    assert np.abs(err).max() < 5.0
    assert np.max(np.abs(rng - rng_true)) < 5.0
    # geometry check
    sta_ecef = geodetic_to_ecef(35.7, 51.4, 1200.0)
    az_el = []
    for jd_t, fr_t in zip(jd_all[::500], fr_all[::500], strict=False):
        r_teme, v_teme = model.state_teme(model.base_elements, jd_t, fr_t)
        r_ecef, _ = teme_to_ecef_state(r_teme, v_teme, jd_t + fr_t)
        az_el.append(az_el_range(r_ecef, sta_ecef, 35.7, 51.4)[1])
    assert np.max(np.abs(np.array(az_el) - el_true[::500])) < 0.1

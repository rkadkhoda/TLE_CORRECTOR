import dataclasses

import numpy as np
from sgp4.api import Satrec

from tlecorrect.sgp4wrap import OrbitElements, OrbitModel
from tlecorrect.timeutil import jd_split, parse_iso
from tlecorrect.tle import parse_tle

L1 = "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999"
L2 = "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912"


def test_rebuild_matches_reference_satrec():
    t = parse_tle(L1 + "\n" + L2)
    model = OrbitModel(t)
    ref = Satrec.twoline2rv(t.line1, t.line2)
    rebuilt = model.make_satrec(model.base_elements)
    jd, fr = jd_split(parse_iso("2026-10-21T18:22:00"))
    e1, r1, v1 = ref.sgp4(jd, fr)
    e2, r2, v2 = rebuilt.sgp4(jd, fr)
    assert e1 == e2 == 0
    assert np.linalg.norm(np.array(r1) - np.array(r2)) * 1000.0 < 1.0


def test_continuous_elements_avoid_tle_rounding():
    """Changing n by a sub-TLE-precision amount must change the state."""
    t = parse_tle(L1 + "\n" + L2)
    model = OrbitModel(t)
    jd, fr = jd_split(parse_iso("2026-10-21T18:22:00"))
    base = model.state_teme(model.base_elements, jd, fr)[0]
    el2 = dataclasses.replace(model.base_elements, n=model.base_elements.n + 1e-9)
    alt = model.state_teme(el2, jd, fr)[0]
    assert np.linalg.norm(base - alt) > 1e-5
    # and the TLE text would round this away (8 decimals only)
    t2 = model.to_tle(el2)
    assert t2.n == t.n


def test_to_tle_preserves_identity_and_epoch():
    t = parse_tle(L1 + "\n" + L2)
    model = OrbitModel(t)
    el = OrbitElements(n=14.13048504, ecc=t.ecc, i=t.i, raan=2.4525, argp=t.argp, M=320.5880)
    t2 = model.to_tle(el)
    assert t2.satnum == t.satnum
    assert t2.epoch == t.epoch
    assert t2.line1 == t.line1


def test_doppler_sign_positive_approaching(golden_dir):
    """A receding pass yields negative Doppler; approaching yields positive."""
    from tlecorrect.timeutil import jd_split

    t = parse_tle(open(golden_dir / "out" / "true.tle").read())
    model = OrbitModel(t)
    sta = type("S", (), {"lat_deg": 35.7, "lon_deg": 51.4, "alt_m": 1200.0})()
    jd1, fr1 = jd_split(parse_iso("2026-10-21T18:16:40"))
    jd2, fr2 = jd_split(parse_iso("2026-10-21T18:27:30"))
    fd1 = model.doppler(model.base_elements, sta, np.array([jd1]), np.array([fr1]), 137.1e6)[0]
    fd2 = model.doppler(model.base_elements, sta, np.array([jd2]), np.array([fr2]), 137.1e6)[0]
    assert fd1 > 2000.0
    assert fd2 < -2000.0

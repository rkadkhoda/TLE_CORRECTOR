import numpy as np

from tlecorrect.generator import SignalSpec, generate_iq, truth_series
from tlecorrect.tle import parse_tle

L1 = "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999"
L2 = "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912"


def _sta():
    return type("S", (), {"lat_deg": 35.7, "lon_deg": 51.4, "alt_m": 1200.0})()


def test_tone_baseband_equals_doppler_minus_offset(golden_dir):
    t = parse_tle(open(golden_dir / "out" / "true.tle").read())
    import datetime as dt

    start = dt.datetime(2026, 10, 21, 18, 22, 0, tzinfo=dt.UTC)
    fc, offset, fs = 137.1e6, 5000.0, 24000.0
    g = generate_iq(t, _sta(), start, 1.0, fs, fc, offset_hz=offset,
                    signal=SignalSpec(type="tone"), snr_db=40.0, seed=0)
    ph = np.unwrap(np.angle(g.iq))
    inst = np.diff(ph) / (2 * np.pi) * fs
    tr = truth_series(t, _sta(), start, 1.0, fc, rate_hz=1000.0)
    expected = np.interp(g.times_s[1:], tr["t"], tr["doppler_hz"]) - offset
    # single-sample phase differences are noisy; compare the block means
    assert abs(inst.mean() - expected.mean()) < 3.0


def test_fm_mean_frequency_equals_doppler_minus_offset(golden_dir):
    t = parse_tle(open(golden_dir / "out" / "true.tle").read())
    import datetime as dt

    start = dt.datetime(2026, 10, 21, 18, 16, 39, tzinfo=dt.UTC)
    fc, offset, fs = 137.1e6, 5000.0, 24000.0
    # exactly one second of a 1000 Hz modulation => integer periods
    g = generate_iq(t, _sta(), start, 1.0, fs, fc, offset_hz=offset,
                    signal=SignalSpec(type="fm", deviation_hz=3000.0, mod_freq_hz=1000.0),
                    snr_db=40.0, seed=0)
    ph = np.unwrap(np.angle(g.iq))
    mean_f = np.mean(np.diff(ph) / (2 * np.pi) * fs)
    tr = truth_series(t, _sta(), start, 1.0, fc, rate_hz=10.0)
    expected = tr["doppler_hz"].mean() - offset
    assert abs(mean_f - expected) < 5.0


def test_snr_scaling():
    t = parse_tle(L1 + "\n" + L2)
    import datetime as dt

    start = dt.datetime(2026, 10, 21, 18, 20, 0, tzinfo=dt.UTC)
    g20 = generate_iq(t, _sta(), start, 0.5, 24000.0, 137.1e6, 5000.0,
                      snr_db=20.0, seed=1)
    g0 = generate_iq(t, _sta(), start, 0.5, 24000.0, 137.1e6, 5000.0,
                     snr_db=0.0, seed=1)
    p20 = np.mean(np.abs(g20.iq) ** 2)
    p0 = np.mean(np.abs(g0.iq) ** 2)
    assert p0 > p20  # more noise at lower SNR

import datetime as dt

import numpy as np
import pytest

from tlecorrect.extractor import (
    AutoExtractor,
    PhaseSlopeExtractor,
    STFTRidgeExtractor,
    fbb_to_fd,
    make_extractor,
    spectrogram,
)
from tlecorrect.generator import SignalSpec, generate_iq, truth_series
from tlecorrect.tle import parse_tle

FC = 137.1e6
FLO = 137.105e6
START = dt.datetime(2026, 10, 21, 18, 20, 0, tzinfo=dt.UTC)


def _tle():
    return parse_tle(
        "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999\n"
        "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912\n"
    )


def _sta():
    return type("S", (), {"lat_deg": 35.7, "lon_deg": 51.4, "alt_m": 1200.0})()


def test_fbb_to_fd_convention():
    assert fbb_to_fd(-2000.0, FLO, FC) == pytest.approx(-2000.0 + 5000.0)
    assert fbb_to_fd(0.0, FLO, FC) == pytest.approx(5000.0)


def test_tone_extraction_accuracy():
    """A pure tone must be recovered to well below the CRLB at 30 dB."""
    tle = _tle()
    g = generate_iq(tle, _sta(), START, 4.0, 24000.0, FC, offset_hz=5000.0,
                    signal=SignalSpec(type="tone"), snr_db=30.0, seed=3)
    tr = truth_series(tle, _sta(), START, 4.0, FC, rate_hz=100.0)
    res = AutoExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="tone",
                                  nfft=2400, hop=600, window_s=0.1, hop_s=0.1,
                                  min_snr_db=-99.0)
    m = [q for q in res.measurements if q.quality == "ok"]
    assert len(m) > 20
    mt = np.array([q.t_rel_s for q in m])
    mf = np.array([q.fd_hz for q in m])
    ref = np.interp(mt, tr["t"], tr["doppler_hz"])
    err = mf - ref
    assert np.sqrt(np.mean(err**2)) < 0.05
    assert abs(err.mean()) < 0.02


def test_fm_carrier_is_recovered():
    """For FM the carrier (mean instantaneous frequency) is measured, not a sideband."""
    tle = _tle()
    g = generate_iq(tle, _sta(), START, 6.0, 24000.0, FC, offset_hz=5000.0,
                    signal=SignalSpec(type="fm", deviation_hz=3000.0, mod_freq_hz=1000.0),
                    snr_db=30.0, seed=4)
    tr = truth_series(tle, _sta(), START, 6.0, FC, rate_hz=100.0)
    res = AutoExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="fm_center",
                                  nfft=2048, hop=512, window_s=1.0, hop_s=1.0,
                                  mod_freq_hz=1000.0, deviation_hz=3000.0,
                                  min_snr_db=-99.0)
    m = [q for q in res.measurements if q.quality == "ok"]
    mt = np.array([q.t_rel_s for q in m])
    mf = np.array([q.fd_hz for q in m])
    ref = np.interp(mt, tr["t"], tr["doppler_hz"])
    err = mf - ref
    assert np.sqrt(np.mean(err**2)) < 0.02
    # the STFT ridge alone locks onto a sideband -> it must be far off
    stft = STFTRidgeExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="fm_center",
                                         nfft=2048, hop=512, min_snr_db=-99.0)
    s = [q for q in stft.measurements if q.quality == "ok"]
    sfd = np.array([q.fd_hz for q in s])
    assert np.min(np.abs(sfd - ref.mean())) > 500.0


def test_time_bookkeeping_is_exact():
    """t_n = t_start + n/fs: the window centre must match the sample grid."""
    tle = _tle()
    g = generate_iq(tle, _sta(), START, 2.0, 24000.0, FC, offset_hz=5000.0,
                    signal=SignalSpec(type="tone"), snr_db=30.0, seed=5)
    res = PhaseSlopeExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="tone",
                                        window_s=0.1, hop_s=0.1, min_snr_db=-99.0)
    for q in res.measurements:
        assert abs((q.time_utc - START).total_seconds() - q.t_rel_s) < 1e-9
    # 0.1 s window centred on the window start + 0.05 s
    assert res.measurements[0].t_rel_s == pytest.approx(0.05, abs=1e-9)
    assert res.measurements[1].t_rel_s == pytest.approx(0.15, abs=1e-9)


def test_crlb_sigma_matches_empirical_scatter():
    """The reported 1-sigma must be within a factor 2 of the observed scatter."""
    tle = _tle()
    g = generate_iq(tle, _sta(), START, 20.0, 24000.0, FC, offset_hz=5000.0,
                    signal=SignalSpec(type="tone"), snr_db=20.0, seed=6)
    tr = truth_series(tle, _sta(), START, 20.0, FC, rate_hz=100.0)
    res = AutoExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="tone",
                                  nfft=2400, hop=600, window_s=0.1, hop_s=0.1,
                                  min_snr_db=-99.0)
    m = [q for q in res.measurements if q.quality == "ok"]
    mt = np.array([q.t_rel_s for q in m])
    mf = np.array([q.fd_hz for q in m])
    sg = np.array([q.sigma_hz for q in m])
    ref = np.interp(mt, tr["t"], tr["doppler_hz"])
    err = mf - ref
    d2 = np.diff(err, 2)
    scatter = float(np.std(d2) / np.sqrt(6.0))
    assert 0.3 < scatter / float(np.median(sg)) < 4.0


def test_aliasing_windows_are_flagged_not_failed():
    """A signal close to the Nyquist edge is flagged, and the run still succeeds."""
    tle = _tle()
    # place the carrier near the Nyquist edge
    g = generate_iq(tle, _sta(), START, 4.0, 24000.0, FC, offset_hz=-9000.0,
                    signal=SignalSpec(type="fm", deviation_hz=3000.0, mod_freq_hz=1000.0),
                    snr_db=20.0, seed=7)
    res = AutoExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="fm_center",
                                  nfft=2048, hop=512, window_s=1.0, hop_s=1.0,
                                  mod_freq_hz=1000.0, deviation_hz=3000.0,
                                  min_snr_db=-99.0)
    flagged = [q for q in res.measurements if q.quality == "flagged"]
    assert flagged, "windows near the Nyquist edge must be flagged"
    assert "aliasing" in flagged[0].reason
    assert len(res.measurements) > 0


def test_low_snr_windows_are_dropped():
    tle = _tle()
    g = generate_iq(tle, _sta(), START, 3.0, 24000.0, FC, offset_hz=5000.0,
                    signal=SignalSpec(type="tone"), snr_db=-10.0, seed=8)
    res = AutoExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="tone",
                                  nfft=2400, hop=600, window_s=0.1, hop_s=0.1,
                                  min_snr_db=0.0)
    assert any(q.quality == "dropped" for q in res.measurements)


def test_spectrogram_keeps_signed_frequencies():
    """A complex baseband signal is not symmetric: the full spectrum is needed."""
    fs = 24000.0
    t = np.arange(24000) / fs
    x = np.exp(2j * np.pi * (-3000.0) * t).astype(np.complex64)
    freqs, _times, power = spectrogram(x, fs, nfft=4096, hop=1024)
    k = int(np.argmax(power[0]))
    assert freqs[k] == pytest.approx(-3000.0, abs=15.0)
    assert freqs.min() < 0 < freqs.max()


def test_make_extractor_rejects_unknown():
    with pytest.raises(ValueError):
        make_extractor("nope")
    for name in ("stft", "phase", "auto"):
        assert make_extractor(name) is not None


def test_extractor_requires_a_long_enough_window():
    tle = _tle()
    g = generate_iq(tle, _sta(), START, 0.01, 24000.0, FC, offset_hz=5000.0,
                    signal=SignalSpec(type="tone"), snr_db=30.0, seed=9)
    with pytest.raises(ValueError):
        PhaseSlopeExtractor().extract(g.iq, 24000.0, START, FC, FLO, mode="tone",
                                      window_s=1e-5, hop_s=1e-5)

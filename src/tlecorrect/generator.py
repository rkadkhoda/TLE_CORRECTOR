"""Test-only in-memory I/Q generator following the satiq conventions.

This is *not* used by the fit. It exists so unit tests and Monte-Carlo sweeps do
not need hundreds of megabytes of files. The conventions follow
``reference/satiq_README.md`` exactly:

* Doppler positive when approaching;
* ``phi(t) = 2*pi*(fc-flo)*t - 2*pi*fc*R(t)/c`` with R at the receive time;
* baseband ``f_bb = (fc + fd) - flo = fd - offset``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from .frames import C_LIGHT, enu_basis, geodetic_to_ecef
from .sgp4wrap import OrbitElements, OrbitModel
from .tle import TLE

__all__ = ["SignalSpec", "generate_iq", "truth_series", "GenResult"]


@dataclass
class SignalSpec:
    type: str = "tone"  # "tone" or "fm"
    deviation_hz: float = 3000.0
    mod_freq_hz: float = 1000.0
    freq_offset_hz: float = 0.0
    amplitude: float = 1.0


@dataclass
class GenResult:
    iq: np.ndarray
    sample_rate_hz: float
    start_utc: dt.datetime
    fc_hz: float
    flo_hz: float
    snr_db: float
    times_s: np.ndarray = field(default_factory=lambda: np.zeros(0))


def generate_iq(
    tle: TLE,
    station,
    start_utc: dt.datetime,
    duration_s: float,
    sample_rate_hz: float,
    fc_hz: float,
    offset_hz: float = 0.0,
    signal: SignalSpec | None = None,
    snr_db: float = 20.0,
    seed: int = 0,
    elements: OrbitElements | None = None,
) -> GenResult:
    """Generate complex64 baseband I/Q of a pass from the true TLE."""
    signal = signal or SignalSpec()
    model = OrbitModel(tle)
    el = elements or model.base_elements
    n = int(round(duration_s * sample_rate_hz))
    t = np.arange(n) / sample_rate_hz

    fc = fc_hz
    flo = fc + offset_hz
    # Range rate on the sample grid, in chunks, then integrate it to phase.
    # Using the same analytic range rate as the fit avoids the interpolation-
    # derivative mismatch of a spline (a few mHz, which biases the orbit).
    from .timeutil import jd_split_many

    rr_t = np.empty(n, dtype=float)
    chunk = 200_000
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        times = [start_utc + dt.timedelta(seconds=float(x))
                 for x in t[start:stop]]
        jd_c, fr_c = jd_split_many(times)
        _, rr_t[start:stop] = model.range_rate(el, station, jd_c, fr_c)
    rng_t = np.empty(n, dtype=float)
    rng_t[0] = 0.0
    if n > 1:
        rng_t[1:] = np.cumsum(0.5 * (rr_t[1:] + rr_t[:-1])) / sample_rate_hz

    two_pi = 2.0 * np.pi
    phi = two_pi * (fc - flo) * t - two_pi * fc * rng_t / C_LIGHT
    if signal.type == "fm" and signal.deviation_hz:
        phase_mod = -(signal.deviation_hz / signal.mod_freq_hz) * np.cos(
            two_pi * signal.mod_freq_hz * t
        )
        phi = phi + phase_mod
    elif signal.freq_offset_hz:
        phi = phi + two_pi * signal.freq_offset_hz * t
    iq = signal.amplitude * np.exp(1j * phi).astype(np.complex64)

    # AWGN, SNR defined over the full sampling bandwidth
    if snr_db is not None and np.isfinite(snr_db):
        rng_obj = np.random.default_rng(seed)
        sigma = signal.amplitude / np.sqrt(2.0 * 10.0 ** (snr_db / 10.0))
        noise = rng_obj.normal(0.0, sigma, n) + 1j * rng_obj.normal(0.0, sigma, n)
        iq = (iq + noise).astype(np.complex64)

    return GenResult(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        start_utc=start_utc,
        fc_hz=fc,
        flo_hz=flo,
        snr_db=snr_db,
        times_s=t,
    )


def truth_series(
    tle: TLE,
    station,
    start_utc: dt.datetime,
    duration_s: float,
    fc_hz: float,
    rate_hz: float = 10.0,
    elements: OrbitElements | None = None,
) -> dict[str, np.ndarray]:
    """Ground-truth time series (az, el, range, range-rate, doppler) at ``rate_hz``."""
    from .frames import teme_to_ecef_states
    from .timeutil import jd_split_many

    model = OrbitModel(tle)
    el = elements or model.base_elements
    n = int(round(duration_s * rate_hz)) + 1
    t = np.arange(n) / rate_hz
    sta_ecef = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
    sat = model.make_satrec(el)
    rng = np.empty(n)
    rr = np.empty(n)
    az = np.empty(n)
    ely = np.empty(n)
    jd_all, fr_all = jd_split_many(
        [start_utc + dt.timedelta(seconds=float(x)) for x in t])
    err, r_all, v_all = sat.sgp4_array(jd_all, fr_all)
    r_ecef, v_ecef = teme_to_ecef_states(r_all, v_all, jd_all, fr_all)
    los = r_ecef * 1000.0 - sta_ecef
    rng = np.linalg.norm(los, axis=1)
    u = los / rng[:, None]
    rr = np.sum(v_ecef * 1000.0 * u, axis=1)
    enu = enu_basis(station.lat_deg, station.lon_deg) @ los.T
    az = np.degrees(np.arctan2(enu[0], enu[1])) % 360.0
    ely = np.degrees(np.arcsin(enu[2] / rng))
    bad = np.asarray(err) != 0
    rng[bad] = rr[bad] = az[bad] = ely[bad] = np.nan
    return {
        "t": t,
        "az_deg": az,
        "el_deg": ely,
        "range_m": rng,
        "range_rate_m_s": rr,
        "doppler_hz": -fc_hz * rr / C_LIGHT,
    }

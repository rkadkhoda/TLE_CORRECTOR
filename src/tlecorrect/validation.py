"""Validation utilities.  This is the ONLY module that reads ``true.tle``.

The extraction and fit code never import this module and never touch the
ground truth.  It is used after a fit only, to report evaluation metrics.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from .frames import rtn_basis
from .sgp4wrap import OrbitElements, OrbitModel
from .timeutil import jd_split_many
from .tle import TLE, parse_tle

__all__ = ["load_true_tle", "compare_elements", "rtn_series", "doppler_rms", "validate"]


def load_true_tle(path: str) -> TLE:
    from .pipeline import read_text_tolerant

    return parse_tle(read_text_tolerant(path))


def compare_elements(fit: OrbitElements, truth: OrbitElements,
                     dt_fit_days: float = 0.0, dt_truth_days: float = 0.0) -> dict:
    """Element differences, with mean anomaly referred to a common time.

    A short-arc Doppler fit cannot separate ``M`` from ``n``: a mean-anomaly
    error at the TLE epoch is largely absorbed by the mean-motion error.  The
    physically meaningful number is therefore ``dM_at_ref_deg``, evaluated at
    the reference time of the pass.  ``dM_at_epoch_deg`` is reported too, but
    it is strongly correlated with ``dn_rev_day`` and must not be read alone.
    """
    m_fit = fit.M + fit.n * 360.0 * dt_fit_days
    m_truth = truth.M + truth.n * 360.0 * dt_truth_days
    return {
        "dM_at_ref_deg": float((m_fit - m_truth + 180.0) % 360.0 - 180.0),
        "dM_at_epoch_deg": float((fit.M - truth.M + 180.0) % 360.0 - 180.0),
        "dn_rev_day": float(fit.n - truth.n),
        "dRAAN_deg": float((fit.raan - truth.raan + 180.0) % 360.0 - 180.0),
        "di_deg": float(fit.i - truth.i),
    }


def _rtn_series(model_fit: OrbitModel, el_fit: OrbitElements,
                model_truth: OrbitModel, el_truth: OrbitElements,
                times: list[dt.datetime]) -> dict[str, np.ndarray]:
    jd, fr = jd_split_many(times)
    err_f, r_f, _ = model_fit.make_satrec(el_fit).sgp4_array(jd, fr)
    err_t, r_t, v_t = model_truth.make_satrec(el_truth).sgp4_array(jd, fr)
    bad = (np.asarray(err_f) != 0) | (np.asarray(err_t) != 0)
    dr = (np.asarray(r_f) - np.asarray(r_t)) * 1000.0
    out = {"R": [], "T": [], "N": []}
    for k in range(len(times)):
        if bad[k]:
            for key in out:
                out[key].append(np.nan)
            continue
        d = rtn_basis(np.asarray(r_t[k]), np.asarray(v_t[k])) @ dr[k]
        out["R"].append(d[0])
        out["T"].append(d[1])
        out["N"].append(d[2])
    return {k: np.asarray(v) for k, v in out.items()}


def rtn_series(model_fit, el_fit, model_truth, el_truth, times):
    return _rtn_series(model_fit, el_fit, model_truth, el_truth, times)


def doppler_rms(model_fit: OrbitModel, el_fit: OrbitElements,
                model_truth: OrbitModel, el_truth: OrbitElements, station,
                times: list[dt.datetime], fc_hz: float) -> float:
    jd, fr = jd_split_many(times)
    fd = model_fit.doppler(el_fit, station, jd, fr, fc_hz)
    fd_t = model_truth.doppler(el_truth, station, jd, fr, fc_hz)
    return float(np.sqrt(np.nanmean((fd - fd_t) ** 2)))


def validate(model: OrbitModel, initial: OrbitElements, fitted: OrbitElements,
             truth_tle: TLE, stations: dict, measurement_sets, fc_hz: float,
             window: tuple[dt.datetime, dt.datetime]) -> dict:
    """Evaluation metrics before/after against ``true.tle`` (evaluation only)."""
    truth_model = OrbitModel(truth_tle)
    truth = truth_model.base_elements

    t0, t1 = window
    n = 200
    times = [t0 + (t1 - t0) * (k / (n - 1)) for k in range(n)]
    before = rtn_series(model, initial, truth_model, truth, times)
    after = rtn_series(model, fitted, truth_model, truth, times)
    t_mid = t0 + (t1 - t0) / 2
    dt_fit = (t_mid - model.base_tle.epoch).total_seconds() / 86400.0
    dt_truth = (t_mid - truth_tle.epoch).total_seconds() / 86400.0
    out = {
        "epoch_mismatch_s": float((model.base_tle.epoch - truth_tle.epoch).total_seconds()),
        "dM_ref_time_utc": t_mid.isoformat(),
        "elements_before": compare_elements(initial, truth, dt_fit, dt_truth),
        "elements_after": compare_elements(fitted, truth, dt_fit, dt_truth),
        "rtn_before_m": {k: {"rms": float(np.sqrt(np.nanmean(v**2))),
                             "max": float(np.nanmax(np.abs(v)))} for k, v in before.items()},
        "rtn_after_m": {k: {"rms": float(np.sqrt(np.nanmean(v**2))),
                            "max": float(np.nanmax(np.abs(v)))} for k, v in after.items()},
        "doppler_rms_before_hz": {},
        "doppler_rms_after_hz": {},
    }
    for ms in measurement_sets:
        station = stations[ms.station_name]
        out["doppler_rms_before_hz"][ms.observation_id] = doppler_rms(
            model, initial, truth_model, truth, station, ms.times_utc, fc_hz)
        out["doppler_rms_after_hz"][ms.observation_id] = doppler_rms(
            model, fitted, truth_model, truth, station, ms.times_utc, fc_hz)
    return out

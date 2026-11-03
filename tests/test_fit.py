"""Least-squares fit tests (C3).

The synthetic measurement sets are built directly from the SGP4 Doppler model
so the test does not depend on the I/Q generator or on the extractor.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from tlecorrect.fit import run_fit
from tlecorrect.measurement import MeasurementSet
from tlecorrect.scenario import FitOptions, Station
from tlecorrect.sgp4wrap import OrbitElements, OrbitModel
from tlecorrect.timeutil import jd_split_many
from tlecorrect.tle import parse_tle
from tlecorrect.validation import compare_elements

FC_HZ = 137.1e6
T0 = dt.datetime(2026, 10, 21, 18, 16, 34, tzinfo=dt.UTC)


def _station() -> Station:
    return Station(name="tehran", lat_deg=35.7, lon_deg=51.4, alt_m=1200.0)


def _measurements(tle_text: str, *, seconds: int = 300, noise_hz: float = 0.005,
                  sigma_hz: float = 0.005, bias_hz: float = 0.0,
                  systematic_hz: float = 0.0, seed: int = 3) -> MeasurementSet:
    model = OrbitModel(parse_tle(tle_text))
    times = [T0 + dt.timedelta(seconds=k) for k in range(seconds)]
    jd, fr = jd_split_many(times)
    fd = model.doppler(model.base_elements, _station(), jd, fr, FC_HZ)
    t_rel = np.arange(seconds, dtype=float)
    rng = np.random.default_rng(seed)
    fd = fd + bias_hz + rng.normal(0.0, noise_hz, seconds)
    if systematic_hz:
        fd = fd + systematic_hz * np.sin(2.0 * np.pi * (t_rel - seconds / 2) / 23.0)
    return MeasurementSet(
        observation_id=0, station_name="tehran", times_utc=times,
        t_rel_s=t_rel, fd_hz=fd, sigma_hz=np.full(seconds, sigma_hz),
    )


def _fit(initial_tle: str, ms: MeasurementSet, **overrides):
    space_model = OrbitModel(parse_tle(initial_tle))
    t_ref = T0 + dt.timedelta(seconds=float(np.median(ms.t_rel_s)))
    opts = FitOptions(**overrides)
    return run_fit(space_model, [ms], {"tehran": _station()}, FC_HZ, t_ref, opts)


def _params(result) -> dict[str, float]:
    return dict(zip(result.param_names, result.theta, strict=True))


def _sigmas(result) -> dict[str, float]:
    return dict(zip(result.param_names, result.sigma, strict=True))


def test_true_initial_stays_at_truth(true_tle_text):
    ms = _measurements(true_tle_text)
    res = _fit(true_tle_text, ms)
    assert res.success
    assert 0.5 < res.reduced_chi2 < 2.0
    for name in ("orbit.M", "orbit.n", "orbit.RAAN", "orbit.i"):
        assert abs(_params(res)[name]) < 5.0 * _sigmas(res)[name]


def test_perturbed_initial_recovers_truth(true_tle_text, pert_tle_text):
    ms = _measurements(true_tle_text, noise_hz=0.004, sigma_hz=0.004)
    res = _fit(pert_tle_text, ms)
    assert res.success
    assert res.rms_hz < 0.02
    assert 0.3 < res.reduced_chi2 < 3.0

    truth_model = OrbitModel(parse_tle(true_tle_text))
    dt_ref_days = (T0 + dt.timedelta(seconds=float(np.median(ms.t_rel_s)))
                   - truth_model.base_tle.epoch).total_seconds() / 86400.0
    cmp = compare_elements(res.elements, truth_model.base_elements,
                           dt_ref_days, dt_ref_days)
    sigmas = _sigmas(res)
    assert abs(cmp["dM_at_ref_deg"]) < 5.0 * sigmas["orbit.M"]
    assert abs(cmp["dn_rev_day"]) < 5.0 * sigmas["orbit.n"]
    assert abs(cmp["dRAAN_deg"]) < 5.0 * sigmas["orbit.RAAN"]
    assert abs(cmp["di_deg"]) < 5.0 * sigmas["orbit.i"]
    # The RTN Jacobian must be linearised at the fitted point, not at the
    # initial TLE; otherwise the along-track sigma is absurdly large.
    assert all(0.0 < v < 50_000.0 for v in res.rtn_sigma["sigma_m"].values())


def test_constant_bias_is_estimated_not_absorbed(true_tle_text):
    ms = _measurements(true_tle_text, bias_hz=0.35)
    res = _fit(true_tle_text, ms)
    assert abs(_params(res)["obs0.bias_hz"] - 0.35) < 5.0 * _sigmas(res)["obs0.bias_hz"]
    for name in ("orbit.M", "orbit.n", "orbit.RAAN", "orbit.i"):
        assert abs(_params(res)[name]) < 5.0 * _sigmas(res)[name]


def test_sigma_floor_calibration_keeps_reduced_chi2_near_one(true_tle_text):
    # The CRLB is deliberately far too small; a systematic ripple mimics the
    # real recordings' unmodelled scatter.
    ms = _measurements(true_tle_text, noise_hz=0.001, sigma_hz=0.0005,
                       systematic_hz=0.02)
    res = _fit(true_tle_text, ms)
    assert res.sigma_floor_hz
    floor = max(res.sigma_floor_hz.values())
    assert floor > 0.005
    assert 0.3 < res.reduced_chi2 < 3.0
    assert np.all(res.measurements[0].sigma_hz >= floor - 1e-12)


def test_compare_elements_uses_reference_epoch():
    tle = parse_tle(
        "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999\n"
        "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912\n"
    )
    truth = OrbitModel(tle).base_elements
    dt_days = 0.25
    dn = 1.0e-3
    # Shift M and n so that the position is identical at the reference time.
    fitted = OrbitElements(n=truth.n + dn, ecc=truth.ecc, i=truth.i,
                           raan=truth.raan, argp=truth.argp,
                           M=(truth.M - dn * 360.0 * dt_days) % 360.0)
    cmp = compare_elements(fitted, truth, dt_days, dt_days)
    assert abs(cmp["dM_at_ref_deg"]) < 1e-9
    assert abs(cmp["dM_at_epoch_deg"] + dn * 360.0 * dt_days) < 1e-9
    assert abs(cmp["dn_rev_day"] - dn) < 1e-12


def test_fit_modules_never_read_ground_truth(project_root: Path):
    forbidden = ("true.tle", "validation")
    for name in ("fit.py", "measurement.py", "extractor.py", "pipeline.py"):
        text = (project_root / "src" / "tlecorrect" / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{name} must stay independent of {token!r}"


def test_cli_fit_end_to_end(golden_out: Path, tmp_path: Path):
    from tlecorrect.cli import main

    out = tmp_path / "cli_fit"
    code = main([
        "fit", str(golden_out / "scenario.json"),
        "--tle", str(golden_out / "true.tle"),
        "--out", str(out),
        "--set", "fit.max_nfev=40",
    ])
    assert code == 0
    assert (out / "report.json").exists()
    assert (out / "corrected.tle").exists()
    assert (out / "measurements.csv").exists()


def test_precalibration_pass_is_reported(true_tle_text):
    ms = _measurements(true_tle_text, noise_hz=0.001, sigma_hz=0.0005,
                       systematic_hz=0.02)
    res = _fit(true_tle_text, ms)
    assert set(res.precalibration) >= {"rms_hz", "reduced_chi2", "cost",
                                       "success", "nfev"}
    assert res.precalibration["reduced_chi2"] > res.reduced_chi2
    assert res.precalibration["nfev"] > 0


def test_profile_sigma_handles_weak_plane_direction(true_tle_text):
    # A purely local Jacobian gives a ~0.1 deg inclination sigma for this
    # pass; the plane direction is nonlinear and must be profiled.
    ms = _measurements(true_tle_text, noise_hz=0.005, sigma_hz=0.005)
    res = _fit(true_tle_text, ms)
    assert 0.0 < _sigmas(res)["orbit.i"] < 0.05
    assert all(np.isfinite(res.sigma))


def test_solver_does_not_stop_at_iteration_zero(true_tle_text, pert_tle_text):
    ms = _measurements(true_tle_text, noise_hz=0.004, sigma_hz=0.004)
    res = _fit(pert_tle_text, ms, calibrate_sigma=False)
    assert res.nfev > 2
    assert res.iteration_log
    assert res.rms_hz < 0.02


def test_truth_cost_is_not_below_fitted_cost(true_tle_text):
    ms = _measurements(true_tle_text, noise_hz=0.004, sigma_hz=0.004)
    res = _fit(true_tle_text, ms, calibrate_sigma=False)
    model = OrbitModel(parse_tle(true_tle_text))
    jd, fr = jd_split_many(ms.times_utc)
    pred = model.doppler(model.base_elements, _station(), jd, fr, FC_HZ)
    r = (ms.fd_hz - pred) / ms.sigma_hz
    truth_cost = 0.5 * float(np.sum(r**2))
    assert res.cost <= truth_cost + 1e-6


def test_zero_measurements_fail_loudly(golden_out, tmp_path):
    from tlecorrect.pipeline import FitError, run_pipeline
    from tlecorrect.scenario import load_scenario_file, scenario_to_config
    from tlecorrect.tle import parse_tle

    cfg = scenario_to_config(load_scenario_file(golden_out / "scenario.json"), golden_out)
    tle = parse_tle((golden_out / "true.tle").read_text())
    empty = MeasurementSet(0, "tehran", [T0], np.array([0.0]), np.array([]), np.array([]))
    with pytest.raises(FitError):
        run_pipeline(cfg, tle, str(tmp_path), measurements=[empty])

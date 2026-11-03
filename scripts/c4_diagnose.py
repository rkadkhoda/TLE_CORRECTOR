"""C4 addendum diagnostics.

Evaluation only: this module reads ``true.tle`` and is never imported by the
fit code.  It answers the addendum questions with *true* errors, not cost:

* element differences truth vs the initial TLE (all six elements + BSTAR);
* truth cost when only (M, n, RAAN, i) are set to truth (the fit subspace) and
  when all elements are truth;
* a diagnostic fit with ``e*cos(w)`` and ``e*sin(w)`` free, and whether the
  Doppler reaches the measured noise floor;
* the RTN displacement of the weakest singular direction over the *whole*
  window (not only at the reference epoch);
* the profile-vs-linear sigma ratio for that weak direction.

A lower cost along a flat valley is never reported as a better orbit: every
interpretation is tied to the true RTN/element error.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import c4_experiments as C

from tlecorrect.fit import ORBIT_NAMES, FitEngine, ParamSpace
from tlecorrect.frames import rtn_basis
from tlecorrect.measurement import DopplerModel
from tlecorrect.sgp4wrap import OrbitElements, OrbitModel
from tlecorrect.timeutil import jd_split_many
from tlecorrect.validation import rtn_series


def _wrap(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def _element_diffs(initial_tle, t_ref: dt.datetime) -> dict:
    ini = OrbitModel(initial_tle).base_elements
    tru = C.C_TRUE_EL
    d = (t_ref - initial_tle.epoch).total_seconds() / 86400.0
    return {
        "dM_epoch_deg": _wrap(ini.M - tru.M),
        "dM_ref_deg": _wrap((ini.M + ini.n * 360.0 * d) - (tru.M + tru.n * 360.0 * d)),
        "dn_rev_day": ini.n - tru.n,
        "dRAAN_deg": _wrap(ini.raan - tru.raan),
        "di_deg": ini.i - tru.i,
        "de": ini.ecc - tru.ecc,
        "dargp_deg": _wrap(ini.argp - tru.argp),
        "dBSTAR": float(initial_tle.bstar - C.TRUE.bstar),
    }


def _noise_floor(ms, station) -> tuple[float, float]:
    jd, fr = jd_split_many(ms.times_utc)
    true_dop = OrbitModel(C.TRUE).doppler(C.C_TRUE_EL, station, jd, fr, C.FC_HZ)
    resid = ms.fd_hz - true_dop
    bias = float(np.mean(resid))
    return float(np.sqrt(np.mean((resid - bias) ** 2))), bias


def _rtn_stats(el, times) -> dict:
    d = rtn_series(OrbitModel(C.TRUE), el, OrbitModel(C.TRUE), C.C_TRUE_EL, times)
    return {k: {"rms": float(np.sqrt(np.nanmean(v**2))),
                "max_abs": float(np.nanmax(np.abs(v)))} for k, v in d.items()}


def _subspace_elements(initial_el: OrbitElements) -> OrbitElements:
    tru = C.C_TRUE_EL
    return OrbitElements(n=tru.n, ecc=initial_el.ecc, i=tru.i,
                         raan=tru.raan, argp=initial_el.argp, M=tru.M)


def _fit_ecosw(ms, station, t_ref, initial_tle, initial_el, noise_floor) -> dict:
    model = OrbitModel(initial_tle)
    dt_ref = (t_ref - initial_tle.epoch).total_seconds() / 86400.0
    jd, fr = jd_split_many(ms.times_utc)
    y = np.asarray(ms.fd_hz, dtype=float)
    sig = np.full(len(y), max(noise_floor, 1e-9))

    def el_of(p):
        mref, n, raan, inc, x, yv = p[:6]
        ecc = math.hypot(x, yv)
        argp = math.degrees(math.atan2(yv, x)) % 360.0
        m_epoch = (mref - n * 360.0 * dt_ref) % 360.0
        return OrbitElements(n=n, ecc=ecc, i=inc, raan=raan % 360.0,
                             argp=argp, M=m_epoch)

    def resid(p):
        pred = model.doppler(el_of(p), station, jd, fr, C.FC_HZ)
        return (y - p[6] - pred) / sig

    p0 = [initial_el.M + initial_el.n * 360.0 * dt_ref, initial_el.n,
          initial_el.raan, initial_el.i,
          initial_el.ecc * math.cos(math.radians(initial_el.argp)),
          initial_el.ecc * math.sin(math.radians(initial_el.argp)), 0.0]
    res = least_squares(resid, p0, method="trf", x_scale="jac", max_nfev=1000)
    el = el_of(res.x)
    pred = model.doppler(el, station, jd, fr, C.FC_HZ)
    rms = float(np.sqrt(np.mean((y - res.x[6] - pred) ** 2)))
    return {
        "success": bool(res.success), "nfev": int(res.nfev),
        "rms_hz": rms, "noise_floor_hz": noise_floor,
        "reaches_floor": bool(rms <= 1.5 * max(noise_floor, 1e-9)),
        "fitted_ecc": float(el.ecc),
        "fitted_ecosw": float(res.x[4]),
        "fitted_esinw": float(res.x[5]),
        "fitted_argp_deg": float(el.argp),
        "rtn_vs_truth": _rtn_stats(el, list(ms.times_utc)),
    }


def _theta_for(space: ParamSpace, elements, t_ref, bias: float = 0.0) -> np.ndarray:
    inc = C._increments(space.base, elements, t_ref, space.orbit_model.base_tle.epoch)
    return space.pack(inc, {0: bias})


def _element_direction(space, theta, v, t_ref):
    """Unit RTN direction at ``t_ref`` and the theta step for 1 km."""
    base_el = space.elements(theta)
    jd, fr = jd_split_many([t_ref])
    a, va = OrbitModel(space.orbit_model.base_tle).state_teme(base_el, jd[0], fr[0])
    basis = rtn_basis(a, va)
    eps = 1e-5
    d = np.zeros(3)
    for j, _name in enumerate(ORBIT_NAMES):
        th = theta.copy()
        th[j] += eps * v[j]
        el = space.elements(th)
        b, _ = OrbitModel(space.orbit_model.base_tle).state_teme(el, jd[0], fr[0])
        d += basis @ (np.asarray(b) - np.asarray(a)) / eps
    norm = float(np.linalg.norm(d))
    rtn = basis @ d / norm if norm > 0 else np.zeros(3)
    step = v / norm if norm > 0 else np.zeros_like(v)
    return rtn, step


def _rtn_of_step(space, theta, dtheta_orbit, times) -> dict:
    v_ext = np.zeros(space.size)
    v_ext[: len(ORBIT_NAMES)] = dtheta_orbit
    model = OrbitModel(space.orbit_model.base_tle)
    el0 = space.elements(theta)
    el1 = space.elements(theta + v_ext)
    d = rtn_series(model, el1, model, el0, times)
    t_ref_idx = len(times) // 2
    return {
        "R": {"rms": float(np.sqrt(np.nanmean(d["R"] ** 2))),
              "max_abs": float(np.nanmax(np.abs(d["R"]))),
              "at_ref": float(d["R"][t_ref_idx])},
        "T": {"rms": float(np.sqrt(np.nanmean(d["T"] ** 2))),
              "max_abs": float(np.nanmax(np.abs(d["T"]))),
              "at_ref": float(d["T"][t_ref_idx])},
        "N": {"rms": float(np.sqrt(np.nanmean(d["N"] ** 2))),
              "max_abs": float(np.nanmax(np.abs(d["N"]))),
              "at_ref": float(d["N"][t_ref_idx])},
    }


def _coherent_along_weak(ms, station, space, th_tru, v, times) -> dict:
    """Systematic residual of the data, projected onto the weak direction.

    ``delta = measured - true Doppler`` (bias removed).  Projecting it onto
    the derivative along the weakest singular vector gives the equivalent
    orbit shift the data would demand even before fitting; a large value for
    the I/Q extraction (and a small one for white-noise surrogates) is the
    direct explanation of the plane bias.
    """
    jd, fr = jd_split_many(ms.times_utc)
    true_dop = OrbitModel(C.TRUE).doppler(C.C_TRUE_EL, station, jd, fr, C.FC_HZ)
    delta = np.asarray(ms.fd_hz, dtype=float) - true_dop
    delta = delta - delta.mean()
    vf = np.zeros(space.size)
    vf[: len(ORBIT_NAMES)] = v
    eps = 1e-3
    model = OrbitModel(space.orbit_model.base_tle)
    dp = (model.doppler(space.elements(th_tru + eps * vf), station, jd, fr, C.FC_HZ)
          - model.doppler(space.elements(th_tru - eps * vf), station, jd, fr, C.FC_HZ))
    dweak = dp / (2.0 * eps)
    den = float(np.dot(dweak, dweak))
    amp = float(np.dot(delta, dweak) / den) if den > 0 else float("nan")
    ac1 = (float(np.dot(delta[:-1], delta[1:]) / np.dot(delta, delta))
           if delta.size > 2 and np.dot(delta, delta) > 0 else float("nan"))
    wa = _rtn_of_step(space, th_tru, amp * v, times)["N"]
    return {"amp_theta": amp, "lag1_autocorr": ac1,
            "equiv_N_rms_m": wa["rms"], "equiv_N_max_m": wa["max_abs"]}


def _profile(space, ms, station, theta_sol, Q, step_scale, alphas):
    n = space.size
    rows = []
    for alpha in alphas:
        fixed = float(alpha / step_scale)

        def resid(coef, fixed=fixed):
            coef_full = np.zeros(n)
            coef_full[: n - 1] = coef
            coef_full[-1] = fixed
            theta = theta_sol + Q @ coef_full
            el = space.elements(theta)
            jd, fr = jd_split_many(ms.times_utc)
            pred = OrbitModel(space.orbit_model.base_tle).doppler(
                el, station, jd, fr, C.FC_HZ)
            bias = space.nuisances(theta).get(0, {}).get("bias_hz", 0.0)
            return (ms.fd_hz - bias - pred) / np.maximum(ms.sigma_hz, 1e-12)

        res = least_squares(resid, np.zeros(n - 1), method="trf",
                            x_scale="jac", max_nfev=200)
        r = resid(res.x)
        rows.append({"alpha_km": alpha, "cost": float(0.5 * np.sum(r**2)),
                     "rms_hz": float(np.sqrt(np.mean((r * ms.sigma_hz) ** 2))),
                     "success": bool(res.success)})
    return rows


def _weak_profile_ratio(space, fit, theta, t_ref):
    """Ratio of the profiled to linear sigma along the weakest direction."""
    eng = FitEngine(space.orbit_model, space, DopplerModel(space.orbit_model, C.FC_HZ),
                    {fit.measurements[0].station_name: C.TEHRAN},
                    fit.measurements, loss="linear", profile_sigma=True)
    J = eng._numerical_jacobian(theta)
    _U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    r = eng._residual(theta)
    target = fit.reduced_chi2 * fit.sigma_inflation**2
    linear = float(np.sqrt(max(target, 0.0)) / sv[-1]) if sv[-1] > 0 else float("nan")
    prof = eng._profile_amplitude(theta, Vt[-1], float(np.sum(r**2)), target, linear)
    return {
        "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "linear_amp": linear, "profile_amp": float(prof),
        "profile_over_linear": float(prof / linear) if linear else float("nan"),
        "singular_values": [float(x) for x in sv],
        "weak_vector_M_n_RAAN_i": [float(x) for x in Vt[-1][: len(ORBIT_NAMES)]],
    }


def diagnose(label, offs, source: str, seed: int) -> dict:
    station = C.TEHRAN
    if source == "golden":
        bundle = C.load_scenario_file(C.GOLD / "scenario.json")
        cfg = C.scenario_to_config(bundle, C.GOLD)
        sets, _ = C.extract_observations(cfg)
        ms = sets[0]
        stations = {s.name: s for s in cfg.stations}
    else:
        if source == "iq":
            ms, _ = C._extract(C.C_TRUE_EL, station, C.T0, 662.0, snr_db=20.0, seed=seed)
        else:
            ms = C._direct_ms(seed, 0.001, t0=C.T0, duration_s=662.0)
        if ms is None:
            return {"error": "no usable measurements"}
        stations = {"tehran": station}
        station = stations.get(ms.station_name, station)

    t_ref = C._t_ref([ms])
    times = list(ms.times_utc)
    initial = C.PERT if offs is None else C._perturb(offs, t_ref)[0]
    initial_el = OrbitModel(initial).base_elements
    fit = C._fit(initial, [ms], stations, t_ref)

    noise, _bias = _noise_floor(ms, station)
    cal_sigma = fit.measurements[0].sigma_hz
    all_truth = C._cost_at(ms, station, C.C_TRUE_TLE, C.C_TRUE_EL,
                           sigma_hz=cal_sigma)
    subspace = C._cost_at(ms, station, C.C_TRUE_TLE, _subspace_elements(initial_el),
                          sigma_hz=cal_sigma)
    ecosw = _fit_ecosw(ms, station, t_ref, initial, initial_el, noise)

    space = ParamSpace(OrbitModel(initial), t_ref, tuple(ORBIT_NAMES),
                       "estimate", False, (0,))
    rtn_ref, step = _element_direction(space, fit.theta, _weak_vec(space, fit), t_ref)
    v = _weak_vec(space, fit)
    v = v / np.linalg.norm(v)
    weak_1km = _rtn_of_step(space, fit.theta, step, times)
    th_tru = _theta_for(space, C.C_TRUE_EL, t_ref)
    alpha = float(np.dot((fit.theta - th_tru)[: len(ORBIT_NAMES)], v))
    weak_actual = _rtn_of_step(space, fit.theta, -alpha * v, times)
    coherent = _coherent_along_weak(ms, station, space, th_tru, v, times)

    fit_rtn = _rtn_stats(fit.elements, times)
    out = {
        "label": label,
        "source": source,
        "seed": seed,
        "n": int(len(fit.residual_hz)),
        "rms_hz": float(fit.rms_hz),
        "noise_floor_hz": noise,
        "element_diffs_truth_vs_initial": _element_diffs(initial, t_ref),
        "truth_cost_all_elements": all_truth,
        "truth_cost_subspace_MnRAANi": subspace,
        "solution_cost": C._cost_at(ms, station, C.C_TRUE_TLE, fit.elements,
                                    sigma_hz=cal_sigma),
        "fit": {
            "dM_ref_deg": C._elem_errors(fit, t_ref)["dM_at_ref_deg"],
            "dn_rev_day": C._elem_errors(fit, t_ref)["dn_rev_day"],
            "dRAAN_deg": C._elem_errors(fit, t_ref)["dRAAN_deg"],
            "di_deg": C._elem_errors(fit, t_ref)["di_deg"],
            "rtn_vs_truth": fit_rtn,
            "rtn_sigma_m": fit.rtn_sigma.get("sigma_m", {}),
            "sigma_inflation": float(fit.sigma_inflation),
        },
        "ecosw_free_fit": ecosw,
        "weak_direction": {
            "rtn_unit_at_ref": [float(x) for x in rtn_ref],
            "theta_step_for_1km": [float(x) for x in step],
            "displacement_1km_at_ref": weak_1km,
            "solution_alpha": alpha,
            "displacement_actual_weak": weak_actual,
            "profile": _profile(space, fit.measurements[0], station, fit.theta,
                                _Q(space, fit), 1.0 / np.linalg.norm(step),
                                (-1.0, -0.5, 0.0, 0.5, 1.0)),
        },
        "profile_ratio": _weak_profile_ratio(space, fit, fit.theta, t_ref),
        "coherent_along_weak": coherent,
    }

    print(f"\n=== {label} [{source}, seed {seed}] ===")
    print(f"  n={out['n']} rms={out['rms_hz']:.6f} Hz  noise floor={noise:.6f} Hz")
    ed = out["element_diffs_truth_vs_initial"]
    print("  initial-truth: dM_ref {dM_ref_deg:+.4f} deg  dn {dn_rev_day:+.3e}  "
          "dRAAN {dRAAN_deg:+.4f}  di {di_deg:+.4f}  de {de:+.2e}  "
          "dargp {dargp_deg:+.4f}  dBSTAR {dBSTAR:+.2e}".format(**ed))
    print(f"  fit-truth   : dM_ref {out['fit']['dM_ref_deg']:+.4f}  "
          f"dn {out['fit']['dn_rev_day']:+.3e}  dRAAN {out['fit']['dRAAN_deg']:+.4f}  "
          f"di {out['fit']['di_deg']:+.4f}")
    print(f"  truth cost  : all-elements rms {all_truth['rms_hz']:.6f} / "
          f"subspace rms {subspace['rms_hz']:.6f} Hz  (noise {noise:.6f})")
    print(f"  e*cosw free : rms {ecosw['rms_hz']:.6f} Hz  "
          f"reaches floor={ecosw['reaches_floor']}  ecc {ecosw['fitted_ecc']:.6g}")
    wr = out["weak_direction"]["displacement_1km_at_ref"]
    print("  weak dir 1 km at ref -> RTN "
          f"{wr['R']['at_ref']:+.0f}/{wr['T']['at_ref']:+.0f}/{wr['N']['at_ref']:+.0f} m; "
          f"over window max|N| {wr['N']['max_abs']:.0f} m, rms N {wr['N']['rms']:.0f} m")
    wa = out["weak_direction"]["displacement_actual_weak"]
    print(f"  solution weak offset alpha={alpha:+.3e}; true N rms {fit_rtn['N']['rms']:.0f} m; "
          f"weak-only N rms {wa['N']['rms']:.0f} m, max {wa['N']['max_abs']:.0f} m")
    pr = out["profile_ratio"]
    print(f"  profile/linear sigma along weak dir = {pr['profile_over_linear']:.3f} "
          f"(cond {pr['condition_number']:.2e})")
    print(f"  coherent residual along weak dir: amp={coherent['amp_theta']:+.3e} "
          f"(equiv N rms {coherent['equiv_N_rms_m']:.0f} m), "
          f"lag1={coherent['lag1_autocorr']:+.3f}")
    return out


def _weak_vec(space, fit):
    eng = FitEngine(space.orbit_model, space, DopplerModel(space.orbit_model, C.FC_HZ),
                    {fit.measurements[0].station_name: C.TEHRAN},
                    fit.measurements, loss="linear")
    J = eng._numerical_jacobian(fit.theta)
    _U, _sv, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt[-1][: len(ORBIT_NAMES)]


def _Q(space, fit):
    eng = FitEngine(space.orbit_model, space, DopplerModel(space.orbit_model, C.FC_HZ),
                    {fit.measurements[0].station_name: C.TEHRAN},
                    fit.measurements, loss="linear")
    J = eng._numerical_jacobian(fit.theta)
    _U, _sv, Vt = np.linalg.svd(J, full_matrices=False)
    return Vt.T


def snr_scaling() -> dict:
    """Confirm the surrogate sigma law against the CRLB 1/sqrt(SNR)."""
    rows = []
    for snr in (30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0):
        surrogate = 0.001 * 10.0 ** ((20.0 - snr) / 20.0)
        crlb_rel = 10.0 ** (-(snr - 20.0) / 20.0)
        rows.append({"snr_db": snr, "surrogate_sigma_hz": surrogate,
                     "crlb_relative_to_20db": crlb_rel,
                     "equal": abs(surrogate - 0.001 * crlb_rel) < 1e-15})
    return {
        "law": "sigma_hz = anchor * 10**((20 - SNR_dB)/20) = anchor * 10**(-SNR_dB/20)",
        "anchor_20db_hz": 0.001,
        "interpretation": "10 dB (one decade of SNR power) multiplies sigma by "
                          "10**(10/20)=sqrt(10)=3.162, exactly the CRLB 1/sqrt(SNR) law",
        "rows": rows,
    }


if __name__ == "__main__":
    cases = [
        ("golden perturbed_scale1 (I/Q recording)", None, "golden", 21),
        ("altitude +3 km (I/Q 20 dB)", (3000.0, 0.0, 0.0), "iq", 24),
        ("combined 10/5/3 km (I/Q 20 dB)", (3000.0, 10000.0, 5000.0), "iq", 25),
        ("altitude +3 km (surrogate 20 dB)", (3000.0, 0.0, 0.0), "surrogate", 24),
        ("combined 10/5/3 km (surrogate 20 dB)", (3000.0, 10000.0, 5000.0),
         "surrogate", 25),
    ]
    out = {"cases": {}, "snr_scaling": snr_scaling()}
    for label, offs, source, seed in cases:
        key = {"golden": "golden",
               "iq": ("altitude_iq" if "altitude" in label else "combined_iq"),
               "surrogate": ("altitude_surrogate" if "altitude" in label
                             else "combined_surrogate")}[source]
        out["cases"][key] = diagnose(label, offs, source, seed)
    p = C.OUT / "c4_addendum.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(C._jsonable(out), indent=2), encoding="utf-8")
    print(f"\nwrote {p}")

"""Report and output writers (``corrected.tle``, ``report.json``, ``report.md``, CSV)."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np

from .fit import FitResult
from .sgp4wrap import OrbitModel
from .timeutil import jd_split_many

__all__ = ["write_outputs", "WriteResult", "format_markdown"]


class WriteResult(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc


def _rounding_effect(orbit: OrbitModel, elements, tle, window) -> dict:
    t0, t1 = window
    n = 100
    times = [t0 + (t1 - t0) * (k / (n - 1)) for k in range(n)]
    jd, fr = jd_split_many(times)
    err_a, r_a, v_a = orbit.make_satrec(elements).sgp4_array(jd, fr)
    rounded = OrbitModel(tle).base_elements
    err_b, r_b, _ = orbit.make_satrec(rounded).sgp4_array(jd, fr)
    bad = (np.asarray(err_a) != 0) | (np.asarray(err_b) != 0)
    dr = (np.asarray(r_a) - np.asarray(r_b)) * 1000.0
    dr[bad] = np.nan
    d = np.sqrt(np.nanmax(np.sum(dr**2, axis=1)))
    return {
        "max_position_effect_m": float(d),
        "note": "position difference between continuous fitted elements and the rounded TLE text",
    }


def _validation_with_extra(model: OrbitModel, initial, fitted, truth_tle, stations,
                           measurement_sets, fc_hz, window) -> dict | None:
    if truth_tle is None:
        return None
    from .validation import validate

    return validate(model, initial, fitted, truth_tle, stations, measurement_sets,
                    fc_hz, window)


def write_outputs(out_dir: str | Path, *, result: FitResult, orbit: OrbitModel,
                  initial_elements, fc_hz: float, config, extraction: dict,
                  truth_tle=None, window=None, before_pred: dict | None = None,
                  station_map: dict | None = None) -> WriteResult:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    corrected = orbit.to_tle(result.elements)
    (out / "corrected.tle").write_text(corrected.as_text(), encoding="utf-8")

    window = window or config.analysis_window()
    rounding = _rounding_effect(orbit, result.elements, corrected, window)
    validation = _validation_with_extra(orbit, initial_elements, result.elements,
                                        truth_tle, station_map or {}, result.measurements,
                                        fc_hz, window)

    before_rms = {}
    if before_pred is not None:
        for ms in result.measurements:
            p = before_pred.get(ms.observation_id)
            if p is None:
                continue
            rr = ms.fd_hz - np.median(ms.fd_hz - p) - p
            before_rms[ms.observation_id] = float(np.sqrt(np.mean(rr**2)))

    report = {
        "tool": "tlecorrect",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "inputs": {
            "initial_tle": orbit.base_tle.as_text(),
            "initial_elements": orbit.base_tle.element_str(),
            "fc_hz": fc_hz,
            "source": getattr(config, "source", ""),
            "observations": [
                {
                    "id": o.id, "station": o.station_name, "data": o.data_path,
                    "start_utc": o.start_utc.isoformat(),
                    "fc_hz": o.fc_hz, "flo_hz": o.flo_hz,
                    "sample_rate_hz": o.sample_rate_hz, "datatype": o.datatype,
                }
                for o in getattr(config, "observations", [])
            ],
            "warnings": list(getattr(config, "warnings", [])),
        },
        "conventions": {
            "doppler_positive": "approaching",
            "doppler": "fd = -fc*dR/dt/c, R at receive time",
            "tle_epoch": "unchanged",
        },
        "extraction": extraction,
        "fit": {
            "free": list(getattr(config.fit, "free", [])),
            "loss": getattr(config.fit, "loss", "soft_l1"),
            "bias_mode": getattr(config.fit, "bias_mode", "estimate"),
            "drift": bool(getattr(config.fit, "drift", False)),
            "tca_shift_s": result.tca_shift_s,
            "sigma_scale": result.sigma_scale,
            "sigma_floor_hz": result.sigma_floor_hz,
            "parameters": {
                name: {
                    "value": float(result.theta[i]),
                    "sigma": float(result.sigma[i]),
                    "unit": result.observability.get("parameters", {}).get(name, {}).get("unit", ""),
                }
                for i, name in enumerate(result.param_names)
            },
            "covariance": result.covariance.tolist(),
            "correlation": result.correlation.tolist(),
            "observability": result.observability,
            "rtn_sigma": result.rtn_sigma,
            "diagnostics": {
                "success": result.success,
                "status": result.status,
                "message": result.message,
                "classification": result.classification,
                "nfev": result.nfev,
                "njev": result.njev,
                "cost": result.cost,
                "reduced_chi2": result.reduced_chi2,
                "rms_hz": result.rms_hz,
                "rms_hz_per_obs": result.rms_hz_per_obs,
                "rms_hz_before_per_obs": before_rms,
                "n_measurements": int(len(result.residual_hz)),
                "n_parameters": len(result.theta),
                "initial_rms_hz": result.initial_cost,
            },
            "warnings": result.warnings,
        },
        "corrected_tle": corrected.as_text(),
        "rounding": rounding,
        "validation": validation,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "report.md").write_text(format_markdown(report), encoding="utf-8")
    _write_csv(out, result, before_pred)
    return WriteResult(report=report, corrected_tle=corrected,
                       paths={"dir": str(out), "report_json": str(out / "report.json"),
                              "report_md": str(out / "report.md"),
                              "corrected_tle": str(out / "corrected.tle")})


def _write_csv(out: Path, result: FitResult, before_pred: dict | None) -> None:
    rows = []
    slices = []
    start = 0
    for ms in result.measurements:
        slices.append((ms, slice(start, start + ms.n)))
        start += ms.n
    for ms, _sl in slices:
        pred = result.pred_hz[ms.observation_id]
        before = (before_pred or {}).get(ms.observation_id)
        for k in range(ms.n):
            rows.append({
                "obs_id": ms.observation_id,
                "station": ms.station_name,
                "time_utc": ms.times_utc[k].isoformat(),
                "t_rel_s": f"{ms.t_rel_s[k]:.6f}",
                "fd_meas_hz": f"{ms.fd_hz[k]:.6f}",
                "sigma_hz": f"{ms.sigma_hz[k]:.6f}",
                "quality": ms.quality[k] if ms.quality else "",
                "fd_pred_before_hz": "" if before is None else f"{before[k]:.6f}",
                "fd_pred_after_hz": f"{pred[k]:.6f}",
                "residual_before_hz": "" if before is None else f"{ms.fd_hz[k] - before[k]:.6f}",
                "residual_after_hz": f"{ms.fd_hz[k] - pred[k]:.6f}",
            })
    with open(out / "measurements.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["obs_id"])
        w.writeheader()
        w.writerows(rows)
    with open(out / "iterations.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["iteration", "cost", "gradient_norm", "step_norm"])
        w.writeheader()
        w.writerows(result.iteration_log or [{"iteration": 0, "cost": result.cost,
                                              "gradient_norm": "", "step_norm": ""}])


def format_markdown(report: dict) -> str:
    fit = report["fit"]
    d = fit["diagnostics"]
    lines = [
        "# tlecorrect report",
        "",
        f"Initial TLE: `{report['inputs']['initial_elements']}`",
        "",
        "## Fit",
        "",
        f"- success: **{d['success']}** ({d['classification']}, status {d['status']})",
        f"- message: {d['message']}",
        f"- TCA/along-track shift used: {fit['tca_shift_s']:.2f} s",
        f"- reduced chi-square: {d['reduced_chi2']:.3f}",
        f"- Doppler residual RMS: {d['rms_hz']:.4f} Hz (before: {d['initial_rms_hz']:.4f} Hz)",
        f"- measurements/parameters: {d['n_measurements']}/{d['n_parameters']}",
        "",
        "| parameter | value | 1-sigma | unit |",
        "|---|---|---|---|",
    ]
    for name, p in fit["parameters"].items():
        lines.append(f"| {name} | {p['value']:.8g} | {p['sigma']:.4g} | {p['unit']} |")
    rtn = fit.get("rtn_sigma", {}).get("sigma_m") if fit.get("rtn_sigma") else None
    if rtn:
        lines += ["", f"RTN 1-sigma at reference: R {rtn['R']:.1f} m, "
                      f"T {rtn['T']:.1f} m, N {rtn['N']:.1f} m"]
    floors = fit.get("sigma_floor_hz") or {}
    if floors:
        txt = ", ".join(f"obs{oid}: {val * 1000:.3f} mHz" for oid, val in floors.items())
        lines += [f"Empirical sigma floor (first-pass residual RMS): {txt}"]
    if fit["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w}" for w in fit["warnings"]]
    val = report.get("validation")
    if val:
        lines += ["", "## Evaluation against true.tle (evaluation only)", ""]
        for key in ("elements_before", "elements_after"):
            lines.append(f"- {key}: " + ", ".join(f"{k}={v:.6g}" for k, v in val[key].items()))
        for key in ("rtn_before_m", "rtn_after_m"):
            lines.append(f"- {key}: " + ", ".join(
                f"{k} rms={v['rms']:.1f} m max={v['max']:.1f} m" for k, v in val[key].items()))
        lines.append("- Doppler RMS before: " + ", ".join(
            f"obs{k}={v:.4f} Hz" for k, v in val["doppler_rms_before_hz"].items()))
        lines.append("- Doppler RMS after: " + ", ".join(
            f"obs{k}={v:.4f} Hz" for k, v in val["doppler_rms_after_hz"].items()))
    lines += ["", f"TLE rounding position effect: {report['rounding']['max_position_effect_m']:.2f} m"]
    return "\n".join(lines) + "\n"

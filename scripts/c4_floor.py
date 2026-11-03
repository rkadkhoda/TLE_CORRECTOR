"""Stage 1 item 3: where does the golden ~0.079 Hz residual floor come from?

Decomposes the floor into (a) extraction error against the generator's truth
CSV, (b) generator-vs-SGP4 model mismatch, (c) window-averaging bias, and
checks the candidate causes: residual spectrum/autocorrelation, window edges,
Nyquist aliasing, FM modulation leakage, light-time.  Writes
``testdata/runs/c4/c4_floor.json`` and prints the table used in the report.

Run:  .\\.venv\\Scripts\\python.exe scripts\\c4_floor.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tlecorrect.extractor import make_extractor  # noqa: E402
from tlecorrect.frames import C_LIGHT  # noqa: E402
from tlecorrect.pipeline import read_text_tolerant  # noqa: E402
from tlecorrect.scenario import (  # noqa: E402
    load_scenario_file,
    scenario_to_config,
)
from tlecorrect.sgp4wrap import OrbitModel  # noqa: E402
from tlecorrect.sigmfio import SigMFRecording  # noqa: E402
from tlecorrect.timeutil import jd_split_many  # noqa: E402
from tlecorrect.tle import parse_tle  # noqa: E402

GOLDEN = ROOT / "testdata" / "golden01" / "out"
OUT = ROOT / "testdata" / "runs" / "c4" / "c4_floor.json"
CACHE = ROOT / "testdata" / "runs" / "c4" / "floor_data.npz"


def _config():
    bundle = load_scenario_file(GOLDEN / "scenario.json")
    return scenario_to_config(bundle, GOLDEN)


def _truth_csv():
    rows = list(csv.DictReader(
        open(GOLDEN / "NOAA19_tehran_20261021T181639_truth.csv", encoding="utf-8")))
    t = np.array([(dt.datetime.fromisoformat(r["iso_time"])
                   - dt.datetime.fromisoformat(rows[0]["iso_time"])).total_seconds()
                  for r in rows])
    return {
        "t": t,
        "el": np.array([float(r["el_deg"]) for r in rows]),
        "range": np.array([float(r["range_m"]) for r in rows]),
        "rr": np.array([float(r["range_rate_m_s"]) for r in rows]),
        "fd": np.array([float(r["doppler_hz"]) for r in rows]),
        "snr": np.array([float(r["snr_db"]) for r in rows]),
        "start": dt.datetime.fromisoformat(rows[0]["iso_time"]),
    }


def _extract(cfg, window_s: float, hop_s: float, method: str = "auto"):
    ext = make_extractor(method)
    obs = cfg.enabled_observations()[0]
    rec = SigMFRecording(obs.meta_path, obs.data_path)
    x = rec.read()
    kw = dict(nfft=cfg.extractor.nfft, window_s=window_s, hop_s=hop_s,
              min_snr_db=cfg.extractor.min_snr_db,
              max_jump_hz=cfg.extractor.max_jump_hz,
              deviation_hz=cfg.extractor.deviation_hz,
              mod_freq_hz=cfg.extractor.mod_freq_hz)
    res = ext.extract(x, rec.sample_rate, rec.datetime, obs.fc_hz, obs.flo_hz,
                      mode=cfg.extractor.mode, **kw)
    used = res.used()
    return {
        "t": np.array([m.t_rel_s for m in used]),
        "fd": np.array([m.fd_hz for m in used]),
        "sigma": np.array([m.sigma_hz for m in used]),
        "snr": np.array([m.snr_db for m in used]),
        "time": [m.time_utc for m in used],
        "n_total": len(res.measurements),
        "n_ok": len(used),
        "n_flag": sum(m.quality == "flagged" for m in res.measurements),
        "n_drop": sum(m.quality == "dropped" for m in res.measurements),
        "reasons": [m.reason for m in res.measurements if m.quality != "ok"][:20],
    }


def _acf(x: np.ndarray, max_lag: int = 20) -> list[float]:
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 0:
        return []
    return [float(np.dot(x[:-k], x[k:]) / den) for k in range(1, max_lag + 1)]


def _psd_peaks(t: np.ndarray, x: np.ndarray):
    """Periodogram of an irregularly sampled residual + its dominant periods."""
    x = x - x.mean()
    grid = np.linspace(1.0 / (t.max() - t.min()), 0.5, 4000)
    # Lomb-Scargle without the scipy dependency: direct DFT is fine at n~500.
    w = 2.0 * np.pi * grid[:, None] * t[None, :]
    p = np.abs(np.exp(-1j * w) @ x) ** 2
    order = np.argsort(p)[-5:][::-1]
    top = [{"period_s": float(1.0 / grid[k]), "power": float(p[k])} for k in order]
    return top


def _window_average_bias(model, el, station, times, fc, half_s):
    """RMS of (mean Doppler over the window - Doppler at the window centre)."""
    jd, fr = jd_split_many(times)
    f0 = model.doppler(el, station, jd, fr, fc)
    n = 9
    offs = np.linspace(-half_s, half_s, n)
    acc = np.zeros_like(f0)
    for o in offs:
        tt = [x + dt.timedelta(seconds=float(o)) for x in times]
        j2, f2 = jd_split_many(tt)
        acc += model.doppler(el, station, j2, f2, fc)
    return float(np.sqrt(np.mean((acc / n - f0) ** 2)))


def main() -> int:
    cfg = _config()
    obs = cfg.enabled_observations()[0]
    truth_tle = parse_tle(read_text_tolerant(GOLDEN / "true.tle"))
    model = OrbitModel(truth_tle)
    station = cfg.station(obs.station_name)
    csvd = _truth_csv()

    if CACHE.exists():
        z = np.load(CACHE)
        base = {k: z[k] for k in ("t", "fd", "sigma", "snr")}
        meta = {"n_total": int(z["n_total"]), "n_ok": int(z["n_ok"]),
                "n_flag": int(z["n_flag"]), "n_drop": int(z["n_drop"])}
        times = [obs.start_utc + dt.timedelta(seconds=float(x)) for x in base["t"]]
    else:
        base = _extract(cfg, cfg.extractor.window_s, cfg.extractor.hop_s)
        times = base.pop("time")
        meta = {k: base.pop(k) for k in ("n_total", "n_ok", "n_flag", "n_drop")}
        np.savez(CACHE, t=base["t"], fd=base["fd"], sigma=base["sigma"],
                 snr=base["snr"], n_total=meta["n_total"], n_ok=meta["n_ok"],
                 n_flag=meta["n_flag"], n_drop=meta["n_drop"])

    # Truth CSV at the measurement times (the CSV starts 5 s before the recording).
    t_csv = np.array([(x - csvd["start"]).total_seconds() for x in times])
    fd_csv = np.interp(t_csv, csvd["t"], csvd["fd"])
    el_csv = np.interp(t_csv, csvd["t"], csvd["el"])
    snr_csv = np.interp(t_csv, csvd["t"], csvd["snr"])

    jd, fr = jd_split_many(times)
    fd_sgp4 = model.doppler(model.base_elements, station, jd, fr, obs.fc_hz)
    # One fixed-point light-time retardation.
    rng, _ = model.range_rate(model.base_elements, station, jd, fr)
    te = [x - dt.timedelta(seconds=float(r) / C_LIGHT) for x, r in zip(times, rng, strict=True)]
    jd2, fr2 = jd_split_many(te)
    fd_sgp4_lt = model.doppler(model.base_elements, station, jd2, fr2, obs.fc_hz)

    ext_err = base["fd"] - fd_csv
    model_err = fd_csv - fd_sgp4
    total = base["fd"] - fd_sgp4

    # Window-averaging bias of the SGP4 model over the 1 s extraction window.
    wb = _window_average_bias(model, model.base_elements, station, times,
                              obs.fc_hz, 0.5 * cfg.extractor.window_s)
    # The same over the used-centre grid: correlated with the residual shape?
    jdm, frm = jd_split_many([x + dt.timedelta(seconds=0.5) for x in times])
    fd_p5 = model.doppler(model.base_elements, station, jdm, frm, obs.fc_hz)

    # Edge / elevation structure of the total residual.
    quart = np.quantile(el_csv, [0.25, 0.5, 0.75])
    quad_rms = []
    lo = -np.inf
    for hi in list(quart) + [np.inf]:
        k = (el_csv > lo) & (el_csv <= hi)
        quad_rms.append({
            "el_lo": float(lo if np.isfinite(lo) else el_csv.min()),
            "el_hi": float(hi if np.isfinite(hi) else el_csv.max()),
            "n": int(k.sum()),
            "rms_meas_minus_sgp4_hz": float(np.sqrt(np.mean(total[k] ** 2))) if k.any() else None,
            "rms_meas_minus_csv_hz": float(np.sqrt(np.mean(ext_err[k] ** 2))) if k.any() else None,
            "rms_csv_minus_sgp4_hz": float(np.sqrt(np.mean(model_err[k] ** 2))) if k.any() else None,
        })
        lo = hi

    # Window-length sensitivity: if FM modulation leaked, a non-integer number
    # of modulation periods would increase the scatter.
    win_tests = {}
    for w, h in ((0.333, 0.333), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0)):
        try:
            r = _extract(cfg, w, h)
            tc = np.array([(x - csvd["start"]).total_seconds() for x in r["time"]])
            e = r["fd"] - np.interp(tc, csvd["t"], csvd["fd"])
            win_tests[f"{w:g}s"] = {
                "n": int(r["n_ok"]),
                "rms_vs_csv_hz": float(np.sqrt(np.mean(e ** 2))),
                "rms_vs_sgp4_hz": float(np.sqrt(np.mean(
                    (r["fd"] - np.interp(tc, csvd["t"], csvd["fd"])
                     + np.interp(tc, csvd["t"], csvd["fd"])
                     - model.doppler(model.base_elements, station,
                                     *jd_split_many(r["time"]), obs.fc_hz)) ** 2))),
            }
        except Exception as exc:  # pragma: no cover - diagnostic
            win_tests[f"{w:g}s"] = {"error": str(exc)}

    nyq = obs.sample_rate_hz / 2.0
    slope = np.gradient(base["fd"], base["t"])
    result = {
        "method": "extraction vs generator truth CSV vs SGP4(true.tle); 1 s windows",
        "n": int(len(base["t"])),
        "extraction": meta,
        "rms": {
            "measurement_minus_csv_hz": float(np.sqrt(np.mean(ext_err ** 2))),
            "csv_minus_sgp4_hz": float(np.sqrt(np.mean(model_err ** 2))),
            "measurement_minus_sgp4_hz": float(np.sqrt(np.mean(total ** 2))),
            "csv_minus_sgp4_lighttime_hz": float(np.sqrt(np.mean((fd_csv - fd_sgp4_lt) ** 2))),
            "meas_minus_sgp4_lighttime_hz": float(np.sqrt(np.mean((base["fd"] - fd_sgp4_lt) ** 2))),
            "window_average_bias_hz": wb,
        },
        "bias": {
            "measurement_minus_csv_hz": float(ext_err.mean()),
            "csv_minus_sgp4_hz": float(model_err.mean()),
        },
        "acf_lag1": _acf(total, 1)[0] if len(total) > 2 else None,
        "acf_lag1_20": _acf(total, 20),
        "spectrum_top": _psd_peaks(base["t"], total),
        "elevation_quartiles": quad_rms,
        "snr_db": {"min": float(snr_csv.min()), "max": float(snr_csv.max()),
                   "median": float(np.median(snr_csv))},
        "aliasing": {
            "max_abs_fd_hz": float(np.max(np.abs(base["fd"]))),
            "nyquist_hz": float(nyq),
            "min_distance_to_nyquist_hz": float(nyq - np.max(np.abs(base["fd"]))),
            "max_abs_slope_hz_s": float(np.max(np.abs(slope))),
            "flagged_windows": meta["n_flag"],
        },
        "window_lengths": win_tests,
        "corr_total_with_model_error": float(np.corrcoef(total, model_err)[0, 1]),
        "corr_total_with_window_avg_bias": float(np.corrcoef(total, fd_p5 - fd_sgp4)[0, 1]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["rms"], indent=2))
    print("acf lag1", result["acf_lag1"])
    print("elevation quartiles", result["elevation_quartiles"])
    print("window lengths", json.dumps(result["window_lengths"], indent=2))
    print("spectrum", result["spectrum_top"])
    print("aliasing", result["aliasing"])
    print("corr(total, model_err)", result["corr_total_with_model_error"])
    print("corr(total, window_avg)", result["corr_total_with_window_avg_bias"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

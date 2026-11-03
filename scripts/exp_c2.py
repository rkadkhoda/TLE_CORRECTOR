"""C2 experiments: Doppler-extractor accuracy, SNR sweep and edge behaviour.

Run:  .venv\\Scripts\\python.exe scripts\\exp_c2.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from tlecorrect.extractor import AutoExtractor, PhaseSlopeExtractor, STFTRidgeExtractor
from tlecorrect.generator import SignalSpec, generate_iq, truth_series
from tlecorrect.sigmfio import SigMFRecording
from tlecorrect.sgp4wrap import OrbitModel
from tlecorrect.timeutil import jd_split_many, parse_iso
from tlecorrect.tle import parse_tle

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "testdata" / "golden01" / "out"
META = GOLD / "NOAA19_tehran_20261021T181639.sigmf-meta"
TRUTH = GOLD / "NOAA19_tehran_20261021T181639_truth.csv"
FC = 137.1e6
FLO = 137.105e6
OFFSET = FLO - FC


def load_truth():
    t, d, snr = [], [], []
    with open(TRUTH) as fh:
        for row in csv.DictReader(fh):
            t.append(parse_iso(row["iso_time"]))
            d.append(float(row["doppler_hz"]))
            snr.append(float(row["snr_db"]))
    return t, np.array(d), np.array(snr)


def golden_run():
    rec = SigMFRecording(META)
    x = rec.read()
    times, truth, truth_snr = load_truth()
    tt = np.array([(t - rec.datetime).total_seconds() for t in times])
    print(f"golden01: {rec.num_samples} samples, {rec.duration:.1f} s, "
          f"start {rec.datetime.isoformat()}")
    # the SGP4 model of the *true* TLE is the reference the fit will use
    tle = parse_tle((GOLD / "true.tle").read_text())
    model = OrbitModel(tle)
    jd_all, fr_all = jd_split_many(times)
    sta = type("S", (), {"lat_deg": 35.7, "lon_deg": 51.4, "alt_m": 1200.0})()
    fd_model = model.doppler(model.base_elements, sta, jd_all, fr_all, FC)
    d_truth = truth - np.interp(tt, tt, fd_model)
    print(f"  truth CSV doppler vs SGP4(true TLE) model: rms {np.sqrt(np.mean(d_truth**2)):.4f} Hz "
          f"max {np.abs(d_truth).max():.4f} Hz")
    out = {}
    for name in ("stft", "phase", "auto"):
        t0 = time.time()
        ex = {"stft": STFTRidgeExtractor, "phase": PhaseSlopeExtractor,
              "auto": AutoExtractor}[name]()
        res = ex.extract(x, rec.sample_rate, rec.datetime, FC, FLO, mode="fm_center",
                         nfft=2048, hop=512, window_s=1.0, hop_s=1.0, min_snr_db=-5.0,
                         mod_freq_hz=1000.0, deviation_hz=3000.0)
        el = time.time() - t0
        m = res.measurements
        keep = [q for q in m if q.quality == "ok"]
        flagged = [q for q in m if q.quality == "flagged"]
        dropped = [q for q in m if q.quality == "dropped"]
        mt = np.array([q.t_rel_s for q in keep])
        mf = np.array([q.fd_hz for q in keep])
        ref = np.interp(mt, tt, truth)
        ref_model = np.interp(mt, tt, fd_model)
        err = mf - ref
        err_model = mf - ref_model
        rms = float(np.sqrt(np.mean(err**2)))
        out[name] = dict(n_ok=len(keep), n_flagged=len(flagged), n_dropped=len(dropped),
                         rms_hz=rms, bias_hz=float(err.mean()),
                         max_hz=float(np.abs(err).max()),
                         rms_vs_model_hz=float(np.sqrt(np.mean(err_model**2))),
                         bias_vs_model_hz=float(err_model.mean()),
                         seconds=el)
        print(f"  {name:5s} n_ok={len(keep):5d} flagged={len(flagged):4d} "
              f"dropped={len(dropped):4d} rms={rms:7.3f} Hz bias={err.mean():7.3f} "
              f"max={np.abs(err).max():7.3f} | vs SGP4 model rms="
              f"{np.sqrt(np.mean(err_model**2)):7.3f} bias={err_model.mean():7.3f} ({el:.1f} s)")
        if name == "auto":
            ft = np.array([q.t_rel_s for q in flagged])
            if ft.size:
                print(f"        flagged t={ft.min():.0f}..{ft.max():.0f} s "
                      f"({ft.size} windows) e.g. {flagged[0].reason}")
            for k in range(6):
                lo, hi = k * len(keep) // 6, (k + 1) * len(keep) // 6
                print(f"        seg{k}: t={mt[lo]:6.0f}..{mt[hi-1]:6.0f}s "
                      f"rms={np.sqrt(np.mean(err[lo:hi]**2)):7.3f} bias={err[lo:hi].mean():7.3f} "
                      f"rms_model={np.sqrt(np.mean(err_model[lo:hi]**2)):7.3f}")
    return out


def snr_sweep(mode="fm_center", window_s=1.0, snrs=(30, 20, 10, 5, 0), seed=0,
              label=None):
    """SNR sweep on the in-memory generator (truth known exactly)."""
    tle = parse_tle((GOLD / "true.tle").read_text())
    sta = type("S", (), {"lat_deg": 35.7, "lon_deg": 51.4, "alt_m": 1200.0})()
    start = dt.datetime(2026, 10, 21, 18, 20, 0, tzinfo=dt.timezone.utc)
    fs = 24000.0
    rows = []
    is_fm = mode == "fm_center"
    for snr in snrs:
        sig = SignalSpec(type="fm" if is_fm else "tone", deviation_hz=3000.0,
                         mod_freq_hz=1000.0)
        g = generate_iq(tle, sta, start, 120.0, fs, FC, offset_hz=OFFSET, signal=sig,
                        snr_db=float(snr), seed=seed)
        tr = truth_series(tle, sta, start, 120.0, FC, rate_hz=100.0)
        ex = AutoExtractor()
        res = ex.extract(g.iq, fs, start, FC, FLO, mode=mode, nfft=2048, hop=512,
                         window_s=window_s, hop_s=window_s, mod_freq_hz=1000.0,
                         deviation_hz=3000.0, min_snr_db=-99.0)
        m = [q for q in res.measurements if q.quality != "dropped"]
        n_ok = sum(q.quality == "ok" for q in m)
        n_flag = sum(q.quality == "flagged" for q in res.measurements)
        n_drop = sum(q.quality == "dropped" for q in res.measurements)
        mt = np.array([q.t_rel_s for q in m])
        mf = np.array([q.fd_hz for q in m])
        ref = np.interp(mt, tr["t"], tr["doppler_hz"])
        err = mf - ref
        sig_m = np.array([q.sigma_hz for q in m])
        # scatter about a local (3-point) median - removes slow systematic trends
        loc = np.array([np.median(err[max(0, k - 1):k + 2]) for k in range(len(err))])
        scatter = float(np.sqrt(np.mean((err - loc) ** 2)))
        rows.append(dict(snr_db=snr, mode=mode, window_s=window_s, n=len(m),
                         n_ok=n_ok, n_flagged=n_flag, n_dropped=n_drop,
                         rms_hz=float(np.sqrt(np.mean(err**2))),
                         bias_hz=float(err.mean()),
                         scatter_hz=scatter,
                         median_sigma=float(np.median(sig_m)),
                         ratio=float(np.sqrt(np.mean((err / np.maximum(sig_m, 1e-12)) ** 2)))))
        print(f"  {label or mode:10s} snr={snr:4d} dB use={len(m):4d} ok={n_ok:4d} "
              f"flag={n_flag:4d} drop={n_drop:4d} rms={rows[-1]['rms_hz']:9.5f} "
              f"bias={rows[-1]['bias_hz']:9.5f} scatter={scatter:9.5f} "
              f"sigma_med={rows[-1]['median_sigma']:9.6f} "
              f"scatter/sigma={scatter / max(rows[-1]['median_sigma'], 1e-12):9.2f}")
    return rows


def main():
    print("=== golden01 (FM tone, 24 kHz, +5 kHz offset) ===")
    g = golden_run()
    print("=== SNR sweep, in-memory generator, fm_center mode (1 s windows) ===")
    s1 = snr_sweep(mode="fm_center", window_s=1.0, snrs=(30, 20, 15, 10, 5, 0))
    print("=== SNR sweep, in-memory generator, tone mode (0.1 s windows) ===")
    s2 = snr_sweep(mode="tone", window_s=0.1, snrs=(30, 20, 15, 10, 5, 0))
    out = ROOT / "testdata" / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / "c2_results.json").write_text(
        json.dumps({"golden": g, "sweep_fm": s1, "sweep_tone": s2}, indent=2),
        encoding="utf-8")
    print(f"written: {out / 'c2_results.json'}")


if __name__ == "__main__":
    main()

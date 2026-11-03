"""End-to-end pipeline: extraction -> measurement sets -> fit -> outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .extractor import make_extractor
from .fit import FitResult, run_fit
from .measurement import DopplerModel, MeasurementSet
from .report import WriteResult, write_outputs
from .scenario import Config, ConfigError
from .sgp4wrap import OrbitModel
from .sigmfio import SigMFRecording
from .tle import TLE, parse_tle

__all__ = ["PipelineResult", "extract_observations", "run_pipeline", "load_tle_text"]


class FitError(RuntimeError):
    """Raised when a run cannot produce a meaningful fit (CLI exits non-zero)."""


@dataclass
class PipelineResult:
    config: Config
    tle: TLE
    orbit: OrbitModel
    fit: FitResult
    written: WriteResult
    extraction: dict


def load_tle_text(path_or_text: str) -> TLE:
    """Load a TLE from a file if the string names an existing file, else parse it."""
    p = Path(path_or_text)
    if p.exists() and p.is_file():
        return parse_tle(read_text_tolerant(p))
    return parse_tle(path_or_text)


def read_text_tolerant(path) -> str:
    """Read a text file written by any tool (UTF-8/UTF-8-BOM/Windows-1252)."""
    data = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extractor_kwargs(cfg: Config) -> dict:
    ext = cfg.extractor
    kw = {
        "nfft": ext.nfft,
        "window_s": ext.window_s,
        "hop_s": ext.hop_s,
        "min_snr_db": ext.min_snr_db,
        "max_jump_hz": ext.max_jump_hz,
        "deviation_hz": ext.deviation_hz,
        "mod_freq_hz": ext.mod_freq_hz,
    }
    return {k: v for k, v in kw.items() if v is not None}


def extract_observations(cfg: Config) -> tuple[list[MeasurementSet], dict]:
    """Run the extractor on every enabled observation."""
    obs_list = cfg.enabled_observations()
    if not obs_list:
        raise ConfigError("no enabled observations")
    extractor = make_extractor(cfg.extractor.method)
    sets: list[MeasurementSet] = []
    info: dict = {"method": cfg.extractor.method, "mode": cfg.extractor.mode,
                  "settings": _extractor_kwargs(cfg), "observations": {}}
    for obs in obs_list:
        rec = SigMFRecording(obs.meta_path, obs.data_path)
        x = rec.read()
        kw = _extractor_kwargs(cfg)
        if cfg.extractor.method == "stft":
            kw.pop("window_s", None)
        res = extractor.extract(x, rec.sample_rate, rec.datetime, obs.fc_hz, obs.flo_hz,
                                mode=cfg.extractor.mode, **kw)
        used = res.used()
        info["observations"][obs.id] = {
            "station": obs.station_name,
            "n_total": len(res.measurements),
            "n_ok": sum(m.quality == "ok" for m in res.measurements),
            "n_flagged": sum(m.quality == "flagged" for m in res.measurements),
            "n_dropped": sum(m.quality == "dropped" for m in res.measurements),
            "duration_s": rec.duration,
            "start_utc": rec.datetime.isoformat() if rec.datetime else None,
            "warnings": res.warnings,
        }
        if not used:
            raise FitError(
                f"observation {obs.id} ({obs.station_name}): no usable Doppler "
                f"measurements ({len(res.measurements)} windows, all dropped)"
            )
        sets.append(MeasurementSet(
            observation_id=obs.id, station_name=obs.station_name,
            times_utc=[m.time_utc for m in used],
            t_rel_s=np.array([m.t_rel_s for m in used]),
            fd_hz=np.array([m.fd_hz for m in used]),
            sigma_hz=np.array([m.sigma_hz for m in used]),
            quality=[m.quality for m in used],
        ))
    return sets, info


def run_pipeline(cfg: Config, tle: TLE, out_dir: str, *, truth_tle: TLE | None = None,
                 extraction: dict | None = None,
                 measurements: list[MeasurementSet] | None = None) -> PipelineResult:
    if not cfg.enabled_observations():
        raise FitError("no enabled observations")
    fcs = {round(o.fc_hz, 3) for o in cfg.enabled_observations()}
    if len(fcs) != 1:
        raise FitError(f"observations use different carrier frequencies {sorted(fcs)}; "
                       "a common transmit carrier is required")
    fc = float(next(iter(fcs)))
    orbit = OrbitModel(tle)
    if measurements is not None:
        sets = measurements
        if extraction is None:
            extraction = {"method": cfg.extractor.method, "mode": cfg.extractor.mode}
    elif extraction is None:
        sets, extraction = extract_observations(cfg)
    else:
        sets, extraction = extract_observations(cfg)
    if not sets or any(ms.n == 0 for ms in sets):
        raise FitError("no usable measurements to fit")

    all_t = [t for ms in sets for t in ms.times_utc]
    t_ref = min(all_t) + (max(all_t) - min(all_t)) / 2.0

    # Predicted Doppler from the initial TLE (baseline for the report).
    doppler = DopplerModel(orbit, fc)
    stations = {s.name: s for s in cfg.stations}
    before_pred = doppler.predict(orbit.base_elements, stations, sets)

    result = run_fit(orbit, sets, stations, fc, t_ref, cfg.fit,
                     light_time=cfg.fit.light_time)
    if result.classification == "too few valid samples":
        raise FitError(f"fit failed: {result.classification} "
                       f"({len(result.residual_hz)} samples for {len(result.theta)} parameters)")
    written = write_outputs(
        out_dir, result=result, orbit=orbit, initial_elements=orbit.base_elements,
        fc_hz=fc, config=cfg, extraction=extraction, truth_tle=truth_tle,
        window=cfg.analysis_window(), before_pred=dict(
            zip([ms.observation_id for ms in sets], before_pred, strict=True)),
        station_map=stations,
    )
    return PipelineResult(config=cfg, tle=tle, orbit=orbit, fit=result,
                          written=written, extraction=extraction)

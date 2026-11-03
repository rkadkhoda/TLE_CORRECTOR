"""Input configuration: scenario.json / manifest.json importers and the validated
``Config`` object shared by the CLI, GUI and fit.

The project is self-contained: scenario and manifest parsing is tolerant (unknown
keys are ignored) and every value is cross-checked against the SigMF metadata.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .sigmfio import SigMFRecording
from .timeutil import parse_iso

__all__ = [
    "Station",
    "Observation",
    "Window",
    "FitOptions",
    "ExtractorOptions",
    "Config",
    "ScenarioBundle",
    "load_scenario_file",
    "scenario_to_config",
    "manifest_to_config",
    "load_config_yaml",
    "save_config_yaml",
    "ConfigError",
]


class ConfigError(Exception):
    """Raised when required inputs are missing or contradictory."""


def _as_utc(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=dt.UTC)
    return parse_iso(str(value))


class Station(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0
    center_offset_hz: float | None = None
    gain_db: float | None = None
    seed: int | None = None


class Window(BaseModel):
    model_config = ConfigDict(extra="ignore")
    start: dt.datetime
    stop: dt.datetime
    manual_aos: dt.datetime | None = None
    manual_los: dt.datetime | None = None

    @field_validator("start", "stop", "manual_aos", "manual_los", mode="before")
    @classmethod
    def _coerce(cls, v):
        if v is None:
            return None
        return _as_utc(v)


class Recording(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file_names: list[str] = Field(default_factory=list)
    station: str
    recording_start_utc: dt.datetime | None = None
    aos: dt.datetime | None = None
    tca: dt.datetime | None = None
    los: dt.datetime | None = None
    carrier_hz: float | None = None
    sample_rate_hz: float | None = None
    datatype: str | None = None
    formats: list[str] = Field(default_factory=list)

    @field_validator("recording_start_utc", "aos", "tca", "los", mode="before")
    @classmethod
    def _coerce(cls, v):
        if v is None:
            return None
        return _as_utc(v)


class PerturbedTLE(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file: str
    scale: float | None = None
    achieved_rtn_km: dict[str, float] | None = None


class TrueTLEInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    file: str | None = None
    line1: str | None = None
    line2: str | None = None
    note: str | None = None


class ScenarioBundle(BaseModel):
    """Tolerant view of a satiq `scenario.json` (version 1.0)."""

    model_config = ConfigDict(extra="ignore")
    schema_version: str | None = None
    generator: dict[str, Any] | None = None
    created_utc: dt.datetime | None = None
    stations: list[Station] = Field(default_factory=list)
    window: Window | None = None
    rf: dict[str, Any] = Field(default_factory=dict)
    signal: dict[str, Any] = Field(default_factory=dict)
    conventions: dict[str, str] = Field(default_factory=dict)
    true_tle: TrueTLEInfo | None = None
    recordings: list[Recording] = Field(default_factory=list)
    perturbed_tles: list[PerturbedTLE] = Field(default_factory=list)

    @field_validator("created_utc", mode="before")
    @classmethod
    def _coerce(cls, v):
        if v is None:
            return None
        return _as_utc(v)


class Observation(BaseModel):
    """One validated observation: one I/Q recording at one station."""

    model_config = ConfigDict(extra="ignore")
    id: int = 0
    station_name: str
    data_path: str
    meta_path: str | None = None
    fc_hz: float
    flo_hz: float
    sample_rate_hz: float
    datatype: str
    start_utc: dt.datetime
    aos: dt.datetime | None = None
    tca: dt.datetime | None = None
    los: dt.datetime | None = None
    num_samples: int | None = None
    enabled: bool = True
    notes: list[str] = Field(default_factory=list)

    @field_validator("start_utc", "aos", "tca", "los", mode="before")
    @classmethod
    def _coerce(cls, v):
        if v is None:
            return None
        return _as_utc(v)

    @property
    def file_path(self) -> Path:
        return Path(self.data_path)


class FitOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    free: list[str] = Field(default_factory=lambda: ["M", "n", "RAAN", "i"])
    loss: Literal["linear", "soft_l1", "huber"] = "soft_l1"
    bias_mode: Literal["estimate", "zero"] = "estimate"
    drift: bool = False
    tikhonov: bool = False
    prior_sigma: dict[str, float] = Field(default_factory=dict)
    x_scale: list[float] | None = None
    f_scale: float = 1.0
    max_nfev: int = 200
    ftol: float = 1e-12
    xtol: float = 1e-12
    gtol: float = 1e-12
    profile_sigma: bool = True
    light_time: bool = False
    calibrate_sigma: bool = True


class ExtractorOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    method: Literal["auto", "stft", "phase"] = "auto"
    mode: Literal["tone", "fm_center"] = "tone"
    window_s: float = 0.1
    hop_s: float | None = None
    nfft: int = 4096
    min_snr_db: float = -5.0
    smooth: int = 5
    max_jump_hz: float = 400.0
    deviation_hz: float | None = None
    mod_freq_hz: float | None = None


class Config(BaseModel):
    """The validated, self-contained configuration used everywhere."""

    model_config = ConfigDict(extra="ignore")
    stations: list[Station] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    window: Window | None = None
    fit: FitOptions = Field(default_factory=FitOptions)
    extractor: ExtractorOptions = Field(default_factory=ExtractorOptions)
    initial_tle_path: str | None = None
    true_tle_path: str | None = None
    source: str = "manual"
    warnings: list[str] = Field(default_factory=list)

    def station(self, name: str) -> Station:
        for s in self.stations:
            if s.name == name:
                return s
        raise ConfigError(f"station {name!r} is not defined")

    def by_id(self, obs_id: int) -> Observation:
        for o in self.observations:
            if o.id == obs_id:
                return o
        raise ConfigError(f"observation id {obs_id} not found")

    def enabled_observations(self) -> list[Observation]:
        return [o for o in self.observations if o.enabled]

    def analysis_window(self) -> tuple[dt.datetime, dt.datetime]:
        if self.window is not None:
            return self.window.start, self.window.stop
        obs = self.enabled_observations()
        if not obs:
            raise ConfigError("no enabled observations and no analysis window")
        starts = [o.start_utc for o in obs]
        stops = []
        for o in obs:
            stop = o.los
            if stop is None and o.num_samples is not None:
                stop = o.start_utc + dt.timedelta(seconds=o.num_samples / o.sample_rate_hz)
            stops.append(stop or o.start_utc)
        return min(starts), max(stops)


# ---------------------------------------------------------------------------
# Importers
# ---------------------------------------------------------------------------
def load_scenario_file(path: str | Path) -> ScenarioBundle:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"scenario file not found: {path}")
    with open(path, encoding="utf-8-sig") as fh:
        data = json.load(fh)
    return ScenarioBundle.model_validate(data)


def _find_recording_files(rec_dir: Path, rec: Recording) -> tuple[Path, Path | None]:
    meta = data = None
    for name in rec.file_names:
        p = rec_dir / name
        if not p.exists():
            p = rec_dir / Path(name).name
        if name.endswith(".sigmf-meta") and p.exists():
            meta = p
        elif name.endswith(".sigmf-data") and p.exists():
            data = p
    if data is None or meta is None:
        # fall back: any sigmf pair in the folder
        metas = sorted(rec_dir.glob("*.sigmf-meta"))
        if len(metas) == 1:
            meta = metas[0]
            data = meta.with_name(meta.name[: -len(".sigmf-meta")] + ".sigmf-data")
    if data is None or meta is None:
        raise ConfigError(
            f"could not locate a SigMF .sigmf-data/.sigmf-meta pair for station "
            f"{rec.station!r} in {rec_dir}"
        )
    return data, meta


def scenario_to_config(scenario: ScenarioBundle, base_dir: str | Path) -> Config:
    """Build a validated Config from a scenario bundle, cross-checked against SigMF."""
    base_dir = Path(base_dir)
    conflicts: list[str] = []
    rf = scenario.rf or {}
    stations = list(scenario.stations)
    station_by_name = {s.name: s for s in stations}
    if not stations:
        conflicts.append("scenario has no stations")
    rf_center_offset = float(rf.get("center_offset_hz", 0.0) or 0.0)
    rf_sdr_center = rf.get("sdr_center_hz")
    fc = rf.get("carrier_hz")
    if fc is None:
        conflicts.append("scenario rf.carrier_hz is missing")

    observations: list[Observation] = []
    for i, rec in enumerate(scenario.recordings):
        station = station_by_name.get(rec.station)
        if station is None:
            conflicts.append(f"recording {i}: unknown station {rec.station!r}")
            continue
        data_path, meta_path = _find_recording_files(base_dir, rec)
        sm = SigMFRecording(meta_path, data_path)
        notes: list[str] = []
        # cross-check sample rate
        if rec.sample_rate_hz is not None and abs(rec.sample_rate_hz - sm.sample_rate) > 1e-6:
            conflicts.append(
                f"recording {i}: sample rate differs scenario={rec.sample_rate_hz} "
                f"sigmf={sm.sample_rate}"
            )
        if rec.datatype is not None and rec.datatype != sm.datatype:
            conflicts.append(
                f"recording {i}: datatype differs scenario={rec.datatype} sigmf={sm.datatype}"
            )
        start = sm.datetime or rec.recording_start_utc
        if start is None:
            conflicts.append(f"recording {i}: no recording start time (SigMF core:datetime)")
            start = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
        if (
            sm.datetime is not None
            and rec.recording_start_utc is not None
            and abs((sm.datetime - rec.recording_start_utc).total_seconds()) > 1e-3
        ):
            conflicts.append(
                f"recording {i}: start time differs scenario={rec.recording_start_utc.isoformat()} "
                f"sigmf={sm.datetime.isoformat()}"
            )
        # SDR center frequency. SigMF core:frequency is authoritative, but the
        # satiq bundle stores the *carrier* there while placing the true SDR
        # center (fc + center_offset) in scenario.rf.sdr_center_hz; when the two
        # disagree we keep the scenario value (corroborated by the recorded
        # baseband) and record the conflict.
        station_offset = (
            station.center_offset_hz if station.center_offset_hz is not None else rf_center_offset
        )
        scenario_center = rf_sdr_center
        if scenario_center is None and fc is not None:
            scenario_center = fc + station_offset
        flo = sm.center_freq
        if flo is None:
            if scenario_center is not None:
                flo = float(scenario_center)
                notes.append("center frequency taken from scenario (SigMF has none)")
            else:
                conflicts.append(f"recording {i}: no SDR center frequency anywhere")
                flo = float("nan")
        elif scenario_center is not None and abs(float(scenario_center) - flo) > 1.0:
            conflicts.append(
                f"recording {i}: SDR center frequency differs sigmf={flo} "
                f"scenario={scenario_center}; using scenario (carrier/offset consistent)"
            )
            flo = float(scenario_center)
        if rec.carrier_hz is not None and fc is not None and abs(rec.carrier_hz - fc) > 1e-6:
            conflicts.append(
                f"recording {i}: carrier differs scenario.rf={fc} recording={rec.carrier_hz}"
            )
        obs_fc = float(rec.carrier_hz if rec.carrier_hz is not None else fc or float("nan"))
        observations.append(
            Observation(
                id=i,
                station_name=rec.station,
                data_path=str(data_path),
                meta_path=str(meta_path),
                fc_hz=obs_fc,
                flo_hz=float(flo),
                sample_rate_hz=sm.sample_rate or float(rec.sample_rate_hz or 0.0),
                datatype=sm.datatype,
                start_utc=start,
                aos=rec.aos,
                tca=rec.tca,
                los=rec.los,
                num_samples=sm.num_samples,
                notes=notes,
            )
        )

    window = scenario.window
    if window is None and observations:
        stops = [o.los for o in observations if o.los]
        starts = [o.start_utc for o in observations]
        if starts:
            window = Window(start=min(starts), stop=max(stops) if stops else max(starts))

    # Choose the extractor mode from the generator signal description.  For an
    # FM source the carrier is the mean instantaneous frequency, so use
    # fm_center; for CW use tone.  The modulation frequency, when known, makes
    # the averaging window an integer number of modulation periods.
    ext = ExtractorOptions()
    if str(scenario.signal.get("type", "")).lower() == "fm":
        ext.mode = "fm_center"
        ext.deviation_hz = _opt_float(scenario.signal.get("deviation_hz"))
        ext.mod_freq_hz = _opt_float(scenario.signal.get("mod_freq_hz"))
        if ext.mod_freq_hz and ext.mod_freq_hz > 0:
            ext.window_s = max(1.0, 1.0 / ext.mod_freq_hz)
        else:
            ext.window_s = 1.0
        ext.hop_s = ext.window_s
        ext.nfft = 2048
    else:
        ext.mode = "tone"

    config = Config(
        stations=stations,
        observations=observations,
        window=window,
        extractor=ext,
        true_tle_path=str(base_dir / "true.tle") if (base_dir / "true.tle").exists() else None,
        source="scenario",
        warnings=conflicts,
    )
    return config


def _opt_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def manifest_to_config(manifest_path: str | Path) -> Config:
    """Import from a raw manifest.json + SigMF metadata (no scenario.json)."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ConfigError(f"manifest file not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8-sig") as fh:
        man = json.load(fh)
    base_dir = manifest_path.parent
    cfg = man.get("config", {})
    rf = cfg.get("rf", {})
    stations = [Station.model_validate(s) for s in cfg.get("stations", [])]
    fc = float(rf.get("carrier_hz"))
    conflicts: list[str] = []
    observations: list[Observation] = []
    for i, res in enumerate(man.get("results", [])):
        if res.get("status") != "ok":
            conflicts.append(f"result {i}: skipped (status={res.get('status')})")
            continue
        data_path = None
        meta_path = None
        for f in res.get("files", []):
            p = (base_dir / f).resolve() if not Path(f).is_absolute() else Path(f)
            if not p.exists():
                p = base_dir / Path(f).name
            if str(f).endswith(".sigmf-meta"):
                meta_path = p
            elif str(f).endswith(".sigmf-data"):
                data_path = p
        if data_path is None or meta_path is None:
            conflicts.append(f"result {i}: SigMF files missing")
            continue
        sm = SigMFRecording(meta_path, data_path)
        observations.append(
            Observation(
                id=i,
                station_name=res.get("station", ""),
                data_path=str(data_path),
                meta_path=str(meta_path),
                fc_hz=fc,
                flo_hz=float(sm.center_freq if sm.center_freq is not None else fc),
                sample_rate_hz=sm.sample_rate,
                datatype=sm.datatype,
                start_utc=sm.datetime,
                aos=_as_utc(res["aos"]) if res.get("aos") else None,
                los=_as_utc(res["los"]) if res.get("los") else None,
                num_samples=sm.num_samples,
            )
        )
    ext = ExtractorOptions()
    source = cfg.get("source", {}) or {}
    if str(source.get("type", "")).lower() == "fm":
        ext.mode = "fm_center"
        ext.deviation_hz = _opt_float(source.get("deviation_hz"))
        ext.mod_freq_hz = _opt_float(source.get("mod_freq_hz"))
        ext.window_s = 1.0
        ext.hop_s = 1.0
        ext.nfft = 2048
    return Config(
        stations=stations,
        observations=observations,
        extractor=ext,
        source="manifest",
        warnings=conflicts,
    )


# ---------------------------------------------------------------------------
# YAML round trip
# ---------------------------------------------------------------------------
def load_config_yaml(path: str | Path) -> Config:
    import yaml

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8-sig") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.model_validate(data)


def save_config_yaml(config: Config, path: str | Path) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(config.model_dump_json())
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)

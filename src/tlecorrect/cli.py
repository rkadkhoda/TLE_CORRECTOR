"""Command-line interface: ``tlecorrect extract|fit|report|scenario``."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from . import __version__
from .extractor import make_extractor
from .pipeline import FitError, load_tle_text, run_pipeline
from .scenario import (
    Config,
    ConfigError,
    load_config_yaml,
    load_scenario_file,
    manifest_to_config,
    save_config_yaml,
    scenario_to_config,
)
from .sigmfio import SigMFRecording

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tlecorrect",
                                description="Estimate a corrected TLE from recorded I/Q Doppler")
    p.add_argument("--version", action="version", version=f"tlecorrect {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="extract Doppler measurements")
    e.add_argument("scenario", help="scenario.json or a YAML config")
    e.add_argument("--out", required=True)
    e.add_argument("--method", choices=["auto", "stft", "phase"])
    e.add_argument("--mode", choices=["tone", "fm_center"])
    e.add_argument("--window-s", type=float)
    e.add_argument("--min-snr-db", type=float)
    e.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    f = sub.add_parser("fit", help="fit a corrected TLE")
    f.add_argument("scenario", help="scenario.json or a YAML config")
    f.add_argument("--tle", required=True, help="initial (erroneous) TLE, file or text")
    f.add_argument("--out", required=True)
    f.add_argument("--free", help="comma-separated free parameters, e.g. M,n,RAAN,i")
    f.add_argument("--true-tle", help="optional true.tle for evaluation only")
    f.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    r = sub.add_parser("report", help="print a report")
    r.add_argument("dir", help="output directory of a fit")

    s = sub.add_parser("scenario", help="build a validated scenario/config file")
    s.add_argument("--manifest", help="import from a satiq manifest.json")
    s.add_argument("--scenario", help="import and validate a scenario.json")
    s.add_argument("--config", help="import a manually written YAML config")
    s.add_argument("--out", required=True, help="output .json or .yaml path")
    s.add_argument("--template", action="store_true", help="write an example YAML template")
    return p


def _load_config(path: str) -> Config:
    p = Path(path)
    if p.suffix.lower() in (".yaml", ".yml"):
        return load_config_yaml(p)
    bundle = load_scenario_file(p)
    return scenario_to_config(bundle, p.parent)


def _coerce(text: str):
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _apply_overrides(cfg: Config, pairs: list[str]) -> None:
    for item in pairs:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        value = _coerce(value)
        parts = key.split(".")
        if len(parts) == 1:
            target, attr = cfg.fit, key
        elif parts[0] in ("fit", "extractor"):
            target, attr = getattr(cfg, parts[0]), parts[1]
        else:
            raise ConfigError(f"unknown --set section {parts[0]!r}")
        if not hasattr(target, attr):
            raise ConfigError(f"unknown setting {key!r}")
        try:
            setattr(target, attr, value)
        except Exception as exc:  # pydantic validation
            raise ConfigError(f"invalid value for {key}: {exc}") from exc


def _cmd_extract(args) -> int:
    cfg = _load_config(args.scenario)
    if args.method:
        cfg.extractor.method = args.method
    if args.mode:
        cfg.extractor.mode = args.mode
    if args.window_s is not None:
        cfg.extractor.window_s = args.window_s
    if args.min_snr_db is not None:
        cfg.extractor.min_snr_db = args.min_snr_db
    _apply_overrides(cfg, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    extractor = make_extractor(cfg.extractor.method)
    kw = {
        "nfft": cfg.extractor.nfft, "window_s": cfg.extractor.window_s,
        "hop_s": cfg.extractor.hop_s, "min_snr_db": cfg.extractor.min_snr_db,
        "max_jump_hz": cfg.extractor.max_jump_hz,
        "deviation_hz": cfg.extractor.deviation_hz, "mod_freq_hz": cfg.extractor.mod_freq_hz,
    }
    kw = {k: v for k, v in kw.items() if v is not None}
    if cfg.extractor.method == "stft":
        kw.pop("window_s", None)
    total = 0
    for obs in cfg.enabled_observations():
        rec = SigMFRecording(obs.meta_path, obs.data_path)
        res = extractor.extract(rec.read(), rec.sample_rate, rec.datetime, obs.fc_hz,
                                obs.flo_hz, mode=cfg.extractor.mode, **kw)
        path = out / f"extract_obs{obs.id}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["time_utc", "t_rel_s", "fd_hz", "sigma_hz", "snr_db",
                        "quality", "reason"])
            for m in res.measurements:
                w.writerow([m.time_utc.isoformat(), f"{m.t_rel_s:.6f}", f"{m.fd_hz:.6f}",
                            f"{m.sigma_hz:.6f}", f"{m.snr_db:.2f}", m.quality, m.reason])
        total += len([m for m in res.measurements if m.quality != "dropped"])
        print(f"obs {obs.id} {obs.station_name}: {len(res.measurements)} windows -> {path}")
    if total == 0:
        print("error: no usable measurements", file=sys.stderr)
        return 2
    return 0


def _cmd_fit(args) -> int:
    cfg = _load_config(args.scenario)
    if args.free:
        cfg.fit.free = [k.strip() for k in args.free.split(",") if k.strip()]
    _apply_overrides(cfg, args.set)
    tle = load_tle_text(args.tle)
    truth = None
    if args.true_tle:
        from .validation import load_true_tle

        truth = load_true_tle(args.true_tle)
    elif cfg.true_tle_path:
        from .validation import load_true_tle

        truth = load_true_tle(cfg.true_tle_path)
    try:
        out = run_pipeline(cfg, tle, args.out, truth_tle=truth)
    except (FitError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    fit = out.fit
    print(f"fit: {fit.classification} (success={fit.success}, status={fit.status})")
    print(f"residual RMS: {fit.rms_hz:.4f} Hz; reduced chi2={fit.reduced_chi2:.3f}; "
          f"n={len(fit.residual_hz)}")
    for name, val, sig in zip(fit.param_names, fit.theta, fit.sigma, strict=True):
        print(f"  {name:16s} {val: .8g} +/- {sig:.3g}")
    rtn = fit.rtn_sigma.get("sigma_m") if fit.rtn_sigma else None
    if rtn:
        print(f"  RTN sigma at ref: R {rtn['R']:.1f} m  T {rtn['T']:.1f} m  N {rtn['N']:.1f} m")
    for w in fit.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"outputs: {args.out}")
    return 0 if fit.success else 3


def _cmd_report(args) -> int:
    path = Path(args.dir) / "report.md"
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    print(path.read_text(encoding="utf-8"))
    return 0


def _cmd_scenario(args) -> int:
    out = Path(args.out)
    if args.template:
        cfg = Config()
        save_config_yaml(cfg, out)
        print(f"template written: {out}")
        return 0
    sources = [args.manifest, args.scenario, args.config]
    if sum(bool(x) for x in sources) != 1:
        print("error: give exactly one of --manifest/--scenario/--config (or --template)",
              file=sys.stderr)
        return 2
    if args.manifest:
        cfg = manifest_to_config(args.manifest)
    elif args.scenario:
        cfg = _load_config(args.scenario)
    else:
        cfg = load_config_yaml(args.config)
    if out.suffix.lower() in (".yaml", ".yml"):
        save_config_yaml(cfg, out)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(json.loads(cfg.model_dump_json()), indent=2), encoding="utf-8")
    for w in cfg.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(f"written: {out} ({len(cfg.observations)} observations)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "extract":
            return _cmd_extract(args)
        if args.command == "fit":
            return _cmd_fit(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "scenario":
            return _cmd_scenario(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

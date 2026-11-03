# satiq — true-TLE satellite I/Q simulator

Generates realistic baseband I/Q of a satellite pass from the **TRUE TLE**
(SGP4), for one or more ground stations on the same absolute UTC timeline.
A separate `perturb` utility creates deliberately perturbed TLEs as initial
guesses for a Doppler-correction program. The simulator itself **never** uses
a perturbed TLE to generate the received signal.

## Math (brief)

- SGP4 (TEME) → ECEF by GMST rotation; ECEF → topocentric ENU → az/el/range.
  Range rate from rotated TEME velocity projected on line of sight.
- Smooth `CubicSpline` of slant range `R(t)` on a 10 Hz coarse grid; per-sample:
  `phi(t) = 2π(fc−flo)·t − 2π·fc·R(t)/c`, `fd = −fc·dR/dt/c` (positive approaching).
- Link budget: `Prx = EIRP − FSPL(R) + Gsat + Ggnd − Lhorizon`; AWGN from kTB+NF
  or fixed-SNR-at-peak; SDR center `flo = fc + offset`.
- Chunked generation keeps memory constant.

### TLE-error formulas (Mode A, physical)

The received I/Q is always generated from the **true** TLE; these errors only
produce *initial-guess* TLE files for an external corrector. RTN basis:
`R = r/|r|` (radial), `N = (r × v)/|r × v|` (cross-track), `T = N × R`
(along-track). All offsets are referenced to a chosen time (epoch / window
start / window middle).

- **Altitude / radial** `dr` → only the semi-major axis (mean motion):
  `a → a + dr`, `dn/n = −(3/2)(dr/a)` (Kepler: `n = sqrt(mu/a³)`). Mode A never
  touches eccentricity or argument of perigee (those belong to Mode B).
- **Along-track** `s` → only the mean anomaly: `dM = s/a` (rad), equivalent to a
  time shift `dt = −dM/n = −s/v` (a positive lead is a negative time shift).
- **Cross-track** `c` → only the RAAN, `dRAAN = c/(a sin i)`, or only the
  inclination, `dInc = c/a`. A RAAN rotation is a plane rotation, so it also
  moves the satellite along-track by `a cos(i) dRAAN`; that side-effect is
  physical and deliberately *not* cancelled by unphysical coupling.
  RAAN is singular near maximum latitude (`|cos u| → 0`); inclination is
  singular near the node (`|sin u| → 0`); the solver raises an explicit error
  naming `u` and suggesting another reference time or `cross_via`.
- Each requested component frees exactly one element (the other deltas stay
  exactly zero); a combined request solves the free elements together as a
  minimum-norm least-squares problem (`scipy.optimize.least_squares`, numerical
  Jacobian, several seeds). The achieved residual is recomputed **after** TLE
  field rounding and checksumming and stored in `perturbation.json`.

## Quickstart

```bash
pip install -e ".[gui,dev]"
python -m satiq.cli passes examples/noaa_137.yaml
python -m satiq.cli estimate examples/noaa_137.yaml
python -m satiq.cli run examples/noaa_137.yaml
python -m satiq.cli predict --tle-file examples/example_noaa.tle --lat 35.7 --lon 51.4 \
  --start 2024-01-01T00:00:00Z --stop 2024-01-01T02:00:00Z --fc 137000000 --out pred.csv
python -m satiq.cli perturb --tle-file examples/example_noaa.tle \
  --start 2024-01-01T00:00:00Z --stop 2024-01-01T02:00:00Z --fc 137000000 --out pert_out
python -m satiq.gui
pytest -q
```

## YAML or manual input (equivalent)

Every parameter has two equivalent entry paths producing the same validated
`SatiqConfig`, so the core never knows which was used:

- **GUI:** *Load YAML* fills all fields (TLE file/pasted-text/NORAD/download,
  stations table + CSV import/export, time, carrier, sample rate, offset,
  signal, EIRP/antennas/noise, min elevation, format/folder, perturbation
  settings). Every field is editable manually from defaults; *Save YAML* writes
  a file the CLI runs unchanged (`satiq run saved.yaml`; pasted TLEs are stored
  as a sidecar `saved.tle`). Invalid fields show inline errors and disable Run.
  Last YAML path / output folder are remembered.
- **CLI:** `satiq run config.yaml --set rf.carrier_hz=437e6 --set rf.output_format=cfile`
  (`signal.*` aliases `rf.*`, `output.format` aliases `rf.output_format`), or no
  YAML at all with flags:
  `satiq run --tle-file X.tle --station "tehran:35.7,51.4,1200,5000,0,0" \
   --start ... --stop ... --carrier 137e6 --fs 8000 --out out_dir`.
  The final merged config is printed before running.
- **Pass selection** (YAML `selection:` block, GUI pass table, or CLI flags)
  reduces data volume *before* generation: `--passes first:2` (first N per
  station), `--passes 1,4,5` (explicit pass numbers), `--min-peak 30`
  (only passes peaking above 30°), `--common-view 2` (only passes seen by ≥2
  stations with overlapping time). Also `min_duration_s`, per-station vs total
  scope, and explicit `indices:` in YAML.

## GUI

Tabs on the left (Satellite & TLE, Stations, Time & Passes, Signal, Channel &
Noise, Output, Perturbed TLEs, Run), plot tabs on the right (Map, Sky,
Elevation/Doppler/Power) in a resizable splitter, collapsible log dock at the
bottom (Copy/Clear, color-coded), status bar (validity, total estimated size,
estimated generation time).

- **Satellite presets** (all NORAD IDs verified against CelesTrak, frequencies
  per NOAA OSPO / AMSAT):

  | Preset | NORAD | Carrier | Source |
  |---|---|---|---|
  | NOAA 15 (APT) | 25338 | 137.62 MHz | verified |
  | NOAA 18 (APT) | 28654 | 137.9125 MHz | verified |
  | NOAA 19 (APT) | 33591 | 137.1 MHz | verified |
  | SO-50 (FM) | 27607 | 436.795 MHz down | verified |
  | AO-85 (FM) | 40967 | 145.98 MHz down | verified |

- **Time window helpers:** UTC ISO fields plus a live local-time line; buttons
  *Now*, *+1 h*, *+3 h*, *+24 h* and *Cover next N passes* (first station, same
  min-elevation mask); the TLE-errors tab shows the equivalent along-track time
  shift `dt = -s/v` in seconds.
- **TLE epoch safety:** the GUI shows the TLE epoch and warns (yellow >7 d,
  red >30 d) when the time window is far from it; "Set window to TLE epoch"
  jumps to epoch+0..3 d, and downloads/pastes default the window to
  max(epoch, now)+24 h.
- **Offline world map** (Natural Earth 110m land, bundled `.npz`, no network):
  coastlines, station markers, per-pass ground tracks (antimeridian-safe),
  satellite position on a time slider, min-elevation footprint circles,
  zoom/pan toolbar; plus a polar sky plot.
- **Theme:** View > Light/Dark/System + toolbar toggle, persisted; matplotlib
  plots follow the theme.
- **Run tab:** parallel jobs, overall progress + Cancel, results table with
  *Open folder* and a quick-look viewer (spectrogram + truth Doppler overlay).
  Output options: multi-format (SigMF/cfile/wav), filename templates
  (`{sat}_{station}_{pass}_{aos:%Y%m%dT%H%M%S}`), per-station/pass subfolders,
  ground-truth on/off + rate, size-warning threshold (default 2 GB).
- Shortcuts Ctrl+O/Ctrl+S, recent-files menu, drag-and-drop YAML/TLE files,
  per-section *Reset to defaults*, auto-suggest sample rate with the reason.

## Standalone GUI (`dist/`)

- `dist/satiq-gui.exe` (108.3 MB, rebuilt 2026-10-20) runs without Python:
  double-click for the GUI, `dist\satiq-gui.exe --selftest` for SigMF + cfile +
  wav generation with schema validation and readback (exit 0 = PASS), or
  `dist\satiq-gui.exe --run config.yaml [--jobs N]` for headless end-to-end runs.
- The bundle embeds `examples/*.tle` + `*.yaml` + the coastline `.npz` read-only
  (PyInstaller `collect_data_files` covers the sigmf JSON schemas, certifi CA
  bundle and soundfile data); all outputs go only to the folder you choose.
- PyInstaller cannot cross-compile: this file targets **Windows 10/11 x64**
  (CPython 3.14). Rebuild elsewhere with `build_gui.sh` (Linux/macOS) or
  `build_gui.bat` (Windows) — each creates a clean venv, installs deps, builds,
  and runs `--selftest` automatically.

TLEs: provide your own file or download from CelesTrak by NORAD ID/name.
On download failure the tool errors clearly — it never invents TLE data.
`examples/*.tle` are offline deterministic fixtures (valid checksums) for tests.

## Layout

```
satiq/ orbit.py sources.py channel.py generate.py writers.py config.py
       batch.py perturb.py predict.py cli.py gui.py gui_form.py resources.py
       estimators.py mapdata.py plot.py tle_utils.py time_utils.py data/
tests/ examples/ docs/screenshots/ dist/satiq-gui.exe satiq-gui.spec
       build_gui.bat build_gui.sh
```

Extension: subclass `Source` (see `TemplateSource`) for BPSK/FSK/AFSK;
`FadingStage/MultipathStage/IonosphereStage/OscillatorDriftStage` are pass-through stubs.

# tlecorrect — status

Standalone TLE corrector: estimates a corrected TLE from recorded I/Q using Doppler.

## Milestones

| # | Milestone | State |
|---|-----------|-------|
| C0 | setup and safety (git, venv, skeleton, STATUS) | done |
| C1 | data layer (SigMF, scenario/manifest import, test generator) | done |
| C2 | Doppler extractor | done |
| C3 | measurement model + least-squares fit (CLI) | done |
| C4 | diagnostics, observability, report, validation 1–6 | pending |
| C5 | GUI (after user go-ahead) | pending |
| C6 | optional packaging | pending |

## Environment

- Project root (and only folder touched by this project): `TLE_corrector2`.
- Python 3.14.7, dedicated venv at `.venv` (interpreter called by explicit path).
- numpy 2.5.4, scipy 1.18.1, sgp4 2.27, pydantic 2.14.1, PyYAML 6.0.3,
  PySide6 6.12.0, matplotlib 3.11.2, sigmf 1.14.0, PyInstaller 6.22.3,
  pytest 9.1.1, ruff 0.17.1.

## Inventory of provided inputs (`testdata/`, `reference/`, `tools/`)

- `testdata/golden01/` — one generator run (read-only).
  - `out/NOAA19_tehran_20261021T181639.sigmf-data` — cf32_le, 24 kHz, 15 888 000 samples (127 MB).
  - `out/...sigmf-meta` — SigMF 1.0.0, `core:datetime` 2026-10-21T18:16:34Z, `core:freq` 137.1e6.
  - `out/..._truth.csv` — 6620 rows @10 Hz (time, az, el, range, range-rate, Doppler, power, SNR).
  - `out/scenario.json`, `out/manifest.json`, `out/true.tle` (evaluation only),
    `out/perturbed_scale1.tle`, `out/perturbation.json`.
  - `pass.yaml`, `noaa19.tle`, `autorun.log`.
- `reference/satiq_README.md`, `reference/scenario_schema.md` — generator conventions.
- `tools/satiq-gui.exe` — generator GUI (optional; can create extra recordings).

## Conventions (copied from the generator docs, confirmed against golden01)

- Doppler is **positive when approaching**.
- `phi(t) = 2*pi*(fc-flo)*t - 2*pi*fc*R(t)/c`, `fd = -fc*dR/dt/c` with R at the receive time.
- baseband `f_bb = (fc + fd) - flo`, hence `fd = f_bb + flo - fc`.
- RTN: R radial (up), T along-track, N cross-track; offsets are perturbed-minus-true.
- Frame chain: SGP4 TEME -> ECEF via GMST; ECEF velocity includes the Earth-rotation
  term; station static in ECEF. Verified: predicted Doppler vs golden01 truth RMS **0.150 Hz**.

## C1 evidence

- `pytest` (31 tests at the end of C1) green; `ruff check src tests` clean.
- Forward physics vs golden01 truth CSV: predicted Doppler RMS **0.150 Hz**, max 2.59 Hz;
  slant range max error 0.1 m; elevation max error < 0.1 deg (tests `test_frames.py`).
- TLE parse -> format round trip is byte-identical (`test_tle.py`).
- Scenario import of golden01: 1 observation, cf32_le, 24 kHz, 15 888 000 samples,
  start 2026-10-21T18:16:34Z, fc 137.1e6.
- Recorded conflict (kept as a warning in the config): SigMF `core:freq` = 137 100 000 Hz
  equals the **carrier**, while the true mixing LO is 137 105 000 Hz
  (`scenario.rf.sdr_center_hz`, offset +5 kHz). The recorded baseband carrier at AOS is
  **-2004.6 Hz** (= fd + fc - flo), confirming LO = 137.105 MHz. Decision: prefer the
  scenario SDR center when it disagrees with SigMF, and warn.
- In-memory generator reproduces the baseband = `fd - offset` convention
  (`test_generator.py`).

## C2 evidence (extractor)

- `pytest` 41 tests green; `ruff check src tests` clean.
- Golden01 (FM tone, 24 kHz, +5 kHz offset, 662 s): auto extractor yields
  **510 ok / 152 flagged / 0 dropped** 1 s windows. Against the SGP4 model of
  `true.tle` the phase/auto RMS is **0.079-0.080 Hz** (median bias +0.003 Hz);
  against the generator truth CSV it is 0.132 Hz. The truth CSV itself differs
  from the SGP4 model by **0.150 Hz RMS / 2.59 Hz max**, so 0.08 Hz is the
  model-recording floor rather than extractor noise.
- Coarse STFT alone reports the +2000 Hz FM sideband (expected in `fm_center`
  mode); phase/auto are the estimators used for measurements.
- In-memory FM sweep (1 s windows, 120 windows): 30/20/15/10 dB give RMS
  0.00203/0.00203/0.00206/0.00215 Hz with a constant +0.0018 Hz bias; the
  per-window scatter matches the CRLB (scatter/sigma 0.92-1.03).
- At 5 dB and 0 dB the phase-slope estimate becomes heavy-tailed (RMS 6.5 and
  27.1 Hz). Those windows are **flagged** by the low-SNR fallback and are
  excluded from the fit; no silent bad measurements are used.
- Fixes found while validating: periodogram noise floor uses `median/ln(2)`;
  STFT SNR is converted to per-sample SNR; `dropped` is not overwritten by
  `flagged`; edge/aliasing flags require >=10 dB; circular-mean frequency
  fallback is used below 7 dB.

## C3 evidence (measurement model + fit CLI)

- `pytest` 48 tests green; `ruff check src tests` clean. New tests cover:
  true-TLE initial stays at truth, perturbed initial recovers truth, constant
  bias is estimated not absorbed, sigma-floor calibration keeps reduced
  chi-square near one, reference-epoch element comparison, static isolation of
  `true.tle`, and an end-to-end CLI fit.
- Golden01, initial = `true.tle`: 510 measurements, 5 free parameters;
  residual RMS **0.0792 Hz**, reduced chi-square **0.968**; recovered
  increments at the pass middle: dM **4.5e-5 deg**, dn **1.7e-5 rev/day**,
  dRAAN **8.0e-4 deg**, di **7.4e-4 deg**. RTN 1-sigma at reference:
  R **636 m**, T **126 m**, N **2962 m**. Validation vs `true.tle`:
  Doppler RMS **0.0021 Hz**, RTN RMS **5.9 / 7.2 / 58.8 m** (R/T/N).
- Golden01, initial = `perturbed_scale1.tle` (dM +5.08 deg, dn -0.0044 rev/day):
  residual RMS **0.0792 Hz**, reduced chi-square **0.969**; validation
  Doppler RMS **4.37 Hz -> 0.0043 Hz**; RTN RMS before/after
  R 1500 -> **161 m**, T 2052 -> **62 m**, N 993 -> **901 m**; RTN 1-sigma
  after the fit: R **774 m**, T **152 m**, N **3617 m**;
  dM at the pass middle before/after **0.0077 -> -0.0049 deg** (within the
  fitted 0.023 deg sigma).
- A short arc cannot separate M from n: the corrected TLE keeps a +0.54 deg
  mean-anomaly offset at the 3-day-old epoch, compensated by dn, while the
  position over the pass is accurate. The report therefore quotes both
  `dM_at_ref_deg` and `dM_at_epoch_deg` and documents the correlation.
- CRLB sigmas are too optimistic for real recordings. Calibration performs a
  first fit, floors every sigma at that observation's residual RMS, then
  refits; floors (and implied scale factors) are written to `report.json`
  and `report.md`. The golden floor is **79.9 mHz**, which is the
  extractor/model error floor measured in C2.
- CLI: `tlecorrect extract|fit|report|scenario` with `--set fit.KEY=VALUE`
  overrides; `fit` writes `corrected.tle`, `report.json`, `report.md`,
  `measurements.csv`, `iterations.csv`. `validation.py` is the only module
  that reads `true.tle`.

## Decisions

- Package layout `src/tlecorrect`, CLI entry point `tlecorrect`.
- Fit parameters: mean motion, mean anomaly, RAAN, inclination (free/fixed in config);
  eccentricity, argument of perigee, BSTAR and epoch stay fixed at the input TLE.
- Linear re-centering of (mean anomaly, mean motion) about the window middle; the
  forward model always rebuilds the SGP4 record from continuous epoch elements via
  `sgp4init` (no TLE-text rounding inside the fit).
- `true.tle` is never read by extraction/fit code; only by validation.
- The fit uses only `quality == "ok"` windows. `flagged` windows (aliasing risk
  or low-SNR fallback) remain in the output for inspection but do not enter
  the least-squares problem; `dropped` windows have no usable signal.
- Reported mean anomaly is referred to the pass middle (`dM_at_ref_deg`);
  the epoch value is also reported but is strongly correlated with dn.

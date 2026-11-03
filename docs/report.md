# tlecorrect — Project Status Report

**Date:** 2026-10-22  
**Scope:** standalone TLE correction from recorded I/Q using Doppler  
**Last commits:** `aa3222b` (C3), `c7b7f38` (C2)  
**Working tree:** C4 work in progress (uncommitted)

---

## 1. Objective

Estimate a corrected TLE from a recorded satellite pass:

1. detect and extract Doppler versus time from the I/Q recording,
2. fit selected TLE elements to the measured Doppler,
3. write the corrected TLE plus a transparent report,
4. validate against truth only in the standalone validation module,
5. document diagnostics, observability and limitations.

Hard constraints: self-contained package, no `satiq` import, no shared/editable
installs, all file access inside this project folder, unit tests and C2/C4
experiments reproducible without the generator.

---

## 2. Milestone status

| Milestone | Description | Status |
|---|---|---|
| C0 | Skeleton, physics core (TLE/SGP4/frames), STATUS | done |
| C1 | Data layer (SigMF, scenario/manifest, config, test generator) | done |
| C2 | Doppler extractor, flags, CRLB, SNR sweeps | done |
| C3 | Measurement model, calibrated fit, CLI, report, validation | done |
| C4 | Diagnostics, observability, report/validation experiments | **in progress** |
| C5 | GUI | pending, not started (needs go-ahead) |
| C6 | Packaging (PyInstaller) | pending |

Test status: **48/48 pytest tests pass**; `ruff check src tests` is clean.

---

## 3. Implemented components

- `src/tlecorrect/tle.py`, `sgp4wrap.py`, `frames.py`, `timeutil.py`:
  in-house TLE parsing, SGP4 wrapper, TEME/ECEF/ENU transforms.
- `src/tlecorrect/sigmf.py`, `scenario.py`, `config.py`: SigMF and scenario
  import, pydantic configuration, YAML loading.
- `src/tlecorrect/generator.py`: in-memory signal generator for tests and C4
  (no dependency on the external scenario generator).
- `src/tlecorrect/extractor.py`: phase-demodulation Doppler, STFT coarse
  search, circular fallback, per-window CRLB sigma and quality flags.
- `src/tlecorrect/measurement.py`: Doppler model with signed convention,
  per-observation constant bias (optional drift), optional light-time.
- `src/tlecorrect/fit.py`: bounded least squares over `M, n, RAAN, i`, robust
  loss, multi-start, empirical sigma-floor calibration, covariance and RTN
  uncertainty, warnings.
- `src/tlecorrect/report.py`: `corrected.tle`, `report.json`, `report.md`,
  `measurements.csv`, `iterations.csv`.
- `src/tlecorrect/validation.py`: truth comparison, reference-epoch anomaly,
  RTN and Doppler errors (only module allowed to read `true.tle`).
- `src/tlecorrect/pipeline.py`, `cli.py`, `__main__.py`: end-to-end pipeline
  and `extract | fit | report | scenario` commands.
- `scripts/c4_experiments.py`: C4 experiment runner (exp1–exp5).

Output policy: only windows with `quality == "ok"` enter the fit. `flagged`
windows (aliasing risk or low-SNR fallback) remain in the extraction outputs
for inspection but are excluded from least squares.

Fit policy: `M, n, RAAN, i` are free; `e`, `argp`, `BSTAR` and the TLE epoch
are fixed at the input TLE. A per-observation constant bias is estimated. The
parameters are re-centred at the measurement midpoint `t_ref`; the TLE epoch
is unchanged. `dM_at_ref_deg` (pass middle) and `dM_at_epoch_deg` (strongly
correlated with `dn_rev_day`) are reported separately because a short arc
cannot separate mean motion from mean anomaly.

Sigma calibration: fit, replace each observation's sigma by its empirical
residual RMS, warm-start and refit, repeated up to four passes. The used
sigma is uniform per observation; `sigma_floor_hz` is reported. This makes
the reduced chi-square near 1.0 on well-behaved data instead of over-weighting
high-SNR windows with an incorrect CRLB shape.

---

## 4. Measured results

### 4.1 C2 — extractor on the golden recording

Golden fixture: 24 kHz, 15,888,000 samples (662 s), centre 137.105 MHz,
true carrier 137.1 MHz, FM tone.

- automatic extraction: **510 ok / 152 flagged / 0 dropped**;
- phase/auto Doppler RMS vs SGP4(true TLE): **0.079–0.080 Hz**;
- vs truth CSV: **0.132 Hz**; truth CSV vs model: 0.150 Hz RMS, 2.59 Hz max;
- FM SNR sweep 30/20/15/10 dB: RMS **0.00203 / 0.00203 / 0.00206 / 0.00215 Hz**,
  constant bias about +0.0018 Hz, scatter/sigma 0.92–1.03;
- at 5/0 dB all windows are flagged by the low-SNR fallback and excluded.

### 4.2 C3 — golden fit (current code)

Fit from `true.tle` (n = 510):

- residual RMS **0.0792 Hz**, reduced chi-square **1.010**;
- `dM_at_ref = 7.1e-5 deg` (±0.0199), `dRAAN = 7.8e-4 deg` (±0.0035),
  `dn = 1.9e-5 rev/day` (±0.0020);
- RTN 1σ: **R 669 m / T 134 m / N 3116 m**;
- validation Doppler RMS **0.00207 Hz**; RTN RMS vs truth
  **R 6.7 m / T 6.9 m / N 60.1 m**.

Fit from `perturbed_scale1.tle` (TLE error about 1.5 km radial, 2 km along,
1 km cross-track):

| quantity | before | after |
|---|---|---|
| Doppler RMS vs truth | 4.373 Hz | **0.00257 Hz** |
| RTN RMS vs truth R | 1500 m | **42 m** |
| RTN RMS vs truth T | 2052 m | **20 m** |
| RTN RMS vs truth N | 993 m | **246 m** |
| `dM_at_ref` | 0.00775 deg | **−0.00138 deg** (±0.0189) |
| `dn` | −0.00442 rev/day | **−1.24e-4 rev/day** (±0.00187) |
| `dRAAN` | −0.00770 deg | **0.00101 deg** (±0.00324) |

The reference-epoch anomaly is resolved: the corrected TLE keeps the original
epoch, so `dM_at_epoch` can be large even when `dM_at_ref` is small.

### 4.3 C4 — diagnostics (partial)

**Exp1 — internal consistency (golden, true TLE initial):** n = 510,
RMS 0.07925 Hz, redchi **1.010**, `dM_at_ref = 7.1e-5 deg` (±0.0199),
`dRAAN = 7.8e-4 deg` (±0.0035), Doppler validation RMS 0.00207 Hz,
floor 0.07925 Hz, RTN 1σ R 669 / T 134 / N 3116 m. **Passes.**

**Exp2 — single-component perturbation recovery** (truth-floor criterion;
golden case uses the real recording, other cases are in-memory 12 kHz FM at
20 dB):

| case | ΔT before→after | ΔR before→after | ΔN before→after | Doppler before→after | success |
|---|---|---|---|---|---|
| golden perturbed | 2201 → **23 m** | 1500 → **42 m** | 1046 → **181 m** | 4.373 → **0.0026 Hz** | yes |
| along-track 10 km | 9997 → **27 m** | 7 → 50 m | 133 → 217 m | 24.713 → **0.0023 Hz** | yes |
| cross-track 5 km | 5 → 30 m | 2 → 65 m | 4978 → **285 m** | 1.474 → **0.0024 Hz** | yes |
| altitude +3 km | 361 → 217 m | 3002 → **858 m** | 42 → 4111 m | 4.534 → **0.0110 Hz** | **no** |
| combined 10/5/3 km | 10022 → **235 m** | 2993 → **938 m** | 4992 → 4499 m | 25.938 → **0.0119 Hz** | **no** |

All fits have reduced chi-square 1.00–1.02. The in-memory truth floor is
about 0.0018–0.0019 Hz (0.079 Hz for golden). The two failing cases meet the
along-track and radial-reduction criteria but leave a residual Doppler error
about 6× the truth floor, because the single pass leaves a coupled radial /
cross-track bias of about 0.9 km radial and 4–4.5 km cross-track. This is a
genuine single-pass observability limitation, not an optimisation failure.

**Exp3 — multi-observation (in-memory, 300 s arcs):**

| configuration | n | `dRAAN` (deg) | `di` (deg) | `dM_ref` (deg) | N 1σ |
|---|---|---|---|---|---|
| 1 pass / 1 station | 106 | −0.00199 | 0.10501 | −0.02577 | 170 m |
| 2 passes / 1 station | 406 | 0.00029 | 0.00079 | −0.00028 | 0.8 m |
| 1 pass / 2 stations | 266 | 2e-5 | 0.00029 | −5e-5 | 7.5 m |

Multiple passes or multiple stations dramatically improve the orbit-plane
(RAAN/inclination) observability and reduce the cross-track sigma by two to
three orders of magnitude.

**Exp4 (basin / SNR sweep):** started, then aborted by the user; no usable
rows were written. **Exp5 (Monte Carlo):** not yet run. Consequently
`docs/C4_results.md` is not yet generated and C4 is not committed.

Figures currently produced: `docs/figures/c4_doppler.png`,
`docs/figures/c4_rtn.png` (combined-perturbation case).

---

## 5. Known issues and limitations

1. **C4 incomplete:** exp4 and exp5 are pending; the C4 report and unit tests
   are not finished; C4 is not committed.
2. **Formal sigmas cover random noise only:** in the in-memory altitude and
   combined cases the formal N sigma is tens of metres while the actual N
   error is about 4 km. The error is a model/observability bias, not random
   scatter. This must be stated explicitly in the C4 report; the C4 Monte
   Carlo is intended to quantify predicted sigma versus empirical scatter.
3. **One-pass cross-track:** a single pass cannot reliably separate RAAN and
   inclination; exp3 quantifies the improvement from a second pass or station.
4. **Golden C3 figures changed** after the latest calibration change (uniform
   empirical sigma and warm-started calibration loop). The numbers above are
   from the current uncommitted code; the previous commit has the older
   numbers.
5. **Uncommitted work:** `src/tlecorrect/fit.py`, `src/tlecorrect/measurement.py`,
   `scripts/c4_experiments.py`, `docs/figures/`, and this report.
6. `run_pipeline` still re-extracts when an extraction is supplied (minor
   inefficiency, not a correctness bug).

---

## 6. Next steps

1. Finish exp4 (basin / SNR sweep) and exp5 (Monte Carlo bias, sign, start
   time) and generate `docs/C4_results.md` with measured numbers and caveats.
2. Add the remaining C4 tests: solver never stops at iteration 0 with a large
   residual; zero-measurement run fails loudly with `FitError`.
3. Update `docs/STATUS.md` with C4 evidence and commit C4.
4. Stop and ask the user before starting C5 (GUI).
5. Then C6 packaging.

---

## 7. Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=testdata\_pytest_tmp
.\.venv\Scripts\ruff.exe check src tests

.\.venv\Scripts\python.exe -m tlecorrect fit testdata\golden01\out\scenario.json `
  --tle testdata\golden01\out\true.tle --out testdata\runs\c3_true

.\.venv\Scripts\python.exe -m tlecorrect fit testdata\golden01\out\scenario.json `
  --tle testdata\golden01\out\perturbed_scale1.tle `
  --true-tle testdata\golden01\out\true.tle --out testdata\runs\c3_perturbed

.\.venv\Scripts\python.exe scripts\c4_experiments.py --only 1,2,3
.\.venv\Scripts\python.exe scripts\c4_experiments.py --only 4,5 --seeds 20
```

Raw C4 results: `testdata/runs/c4/c4_results.json`.

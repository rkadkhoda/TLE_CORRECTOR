# `scenario.json` schema (satiq scenario bundle, version 1.0)

Every `satiq run` writes a **scenario bundle** into the configured output folder,
so that a separate TLE-corrector project can consume exactly what was generated.
The bundle is the hand-off format described in section 4 of the specification.

Files in a bundle:

| file | purpose |
|------|---------|
| `scenario.json` | machine-readable description (this document) |
| `true.tle` | the ground-truth TLE (`# satiq ground-truth TLE — NOT for the corrector`) |
| `perturbed_scale<sc>.tle` | erroneous TLEs (only when `tle_errors.enabled`) |
| `perturbation.json` | requested vs. achieved RTN errors, element deltas, reference time, seed |
| `<recording>.*` | the I/Q files (SigMF `.sigmf-meta`/`.sigmf-data`, `.cfile`, `.wav`) |
| `<recording>_truth.csv` | per-pass ground truth (Doppler, phase, range, SNR, …) |
| `manifest.json` | the raw run manifest (config + per-recording file lists) |

`scenario.json` is validated against `satiq/data/scenario.schema.json`
(JSON Schema draft-07, `additionalProperties: false`) by `tests/test_scenario.py`.
The `schema_version` field is `"1.0"`.

## Top-level fields

- `schema_version` *(string)* — `"1.0"`.
- `generator.name` *(string)* — `"satiq"`.
- `created_utc` *(string, ISO-8601 UTC)* — when the bundle was written.
- `stations` *(array)* — one object per ground station:
  `name`, `lat_deg`, `lon_deg`, `alt_m`, `center_offset_hz`, `gain_db`, `seed`.
- `window` *(object)* — `start`, `stop` (ISO-8601 UTC) and the optional manual
  `manual_aos` / `manual_los` overrides.
- `rf` *(object)* — `carrier_hz` (true carrier `f_c`), `sdr_center_hz`
  (`f_c` + first-station center offset), `center_offset_hz`, `sample_rate_hz`,
  `datatype`, `min_elevation_deg`, `pass_margin_s`.
- `signal` *(object)* — the source parameters (e.g. `{"type": "fm", "deviation_hz": …}`).
- `conventions` *(object)* — see below.
- `true_tle` *(object)* — `file` (`"true.tle"`), `line1`, `line2`, and the
  `note` "ground truth, not for the corrector".
- `recordings` *(array)* — one object **per recording** (not per file):
  `file_names` (basenames of every file written for that recording),
  `station`, `recording_start_utc` (AOS minus the pass margin),
  `aos`, `tca`, `los`, `carrier_hz`, `sample_rate_hz`, `datatype`, `formats`.
- `perturbed_tles` *(array)* — for each scaled variant:
  `file`, `scale`, `achieved_rtn_km` (`{radial_km, along_km, cross_km}`).

## Conventions

```
doppler_positive : "approaching"   # the received frequency is ABOVE f_c when the
                                   # satellite approaches, and falls below when it recedes
phase_model      : "phi(t) = 2*pi*(fc-flo)*t - 2*pi*fc*R(t)/c ; fd = -fc*dR/dt/c"
rtn              : "R radial(up), T along-track, N cross-track; offsets are
                    perturbed-minus-true"
true_tle         : "ground truth, not for the corrector"
```

## Datatypes

`cf32_le` (complex64, the default), `ci16_le` (int16), `ci8` (int8),
`cf32_le` (float32 WAV). The value reflects the on-disk representation.

## Datatype mapping

| `rf.quant` | on-disk |
|------------|---------|
| `complex64` / `float32` | `cf32_le` |
| `int16` | `ci16_le` |
| `int8` | `ci8` |

## Minimal example

```json
{
  "schema_version": "1.0",
  "generator": {"name": "satiq"},
  "created_utc": "2026-10-18T12:00:00+00:00",
  "stations": [{"name": "tehran", "lat_deg": 35.7, "lon_deg": 51.4,
                "alt_m": 1200.0, "center_offset_hz": 5000.0,
                "gain_db": 0.0, "seed": 0}],
  "window": {"start": "2026-10-18T00:00:00+00:00", "stop": "2026-10-19T00:00:00+00:00",
             "manual_aos": null, "manual_los": null},
  "rf": {"carrier_hz": 137000000.0, "sdr_center_hz": 137005000.0,
         "center_offset_hz": 5000.0, "sample_rate_hz": 48000.0,
         "datatype": "cf32_le", "min_elevation_deg": 10.0, "pass_margin_s": 10.0},
  "signal": {"type": "fm", "freq_offset_hz": 0.0, "deviation_hz": 3000.0,
             "mod_freq_hz": 1000.0, "mod": "tone"},
  "conventions": {"doppler_positive": "approaching",
                  "phase_model": "phi(t) = 2*pi*(fc-flo)*t - 2*pi*fc*R(t)/c ; fd = -fc*dR/dt/c",
                  "rtn": "R radial(up), T along-track, N cross-track; offsets are perturbed-minus-true",
                  "true_tle": "ground truth, not for the corrector"},
  "true_tle": {"file": "true.tle", "line1": "1 ...", "line2": "2 ...",
               "note": "ground truth, not for the corrector"},
  "recordings": [],
  "perturbed_tles": []
}
```

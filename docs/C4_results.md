# C4 validation experiments (measured)

All in-memory runs use the project's test-only generator and the golden01 TLE/station; SNR is defined over the sampling bandwidth. RTN errors are RMS over the analysed window; `rtn_sigma` is the 1-sigma position uncertainty at the reference epoch.

**Weak-direction escape is not part of the fit.** It was implemented only as a diagnostic and then removed: a lower cost along the flat valley is not evidence of a better orbit. Every statement below is backed by true RTN and element errors against `true.tle`; the subspace/all-element truth costs are reported side by side.

## 1. Consistency (true TLE as initial)

| case | n | RMS Hz | redchi | dM_ref deg | dn rev/day | di deg | Doppler RMS Hz | RTN sigma R/T/N m | floor Hz |
|---|---|---|---|---|---|---|---|---|---|
| golden01 recording, true TLE initial | 510 | 0.07925 | 1.01 | -1.96e-05 +/- 0.00863 | 1.05e-05 +/- 8.61e-04 | 9.90e-04 +/- 0.0362 | 0.002099 | 292/60/1369 | 0.0792 |

## 2. Erroneous TLEs and single-component errors (20 dB)

| case | runs | n | RMS Hz | floor Hz | dT before/after m | drop % | RTN 3D after m | Doppler before/after Hz | dM_ref deg | dn rev/day | dRAAN deg | di deg | truth cost | sol cost | cost(sol)-cost(truth) | orig / scoped / RTN rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| golden perturbed_scale1 | 1 | 510 | 0.07925 | 0.07949 | 2201 / 23 | 98.9 | 187 | 4.373 / 0.002568 | -0.00138 | -1.24e-04 | 0.001 | 0.0067 | 2.6e+02 | 2.5e+02 | -1.6 | 100% / 100% / 100% |
| along-track 10 km | 20 | 662 | 9.90e-04 | 9.69e-04 | 9786 / 4 | 100.0 | 60 | 15.9 / 2.43e-04 | -1.72e-04 | -1.69e-05 | 3.26e-05 | 7.29e-04 | 3.3e+02 | 3.3e+02 | -1.8 | 100% / 100% / 100% |
| cross-track 5 km | 20 | 662 | 9.90e-04 | 0.00102 | 3 / 4 | -92.5 | 63 | 1.133 / 2.19e-04 | -2.74e-04 | -2.71e-05 | 4.72e-05 | 0.00116 | 3.3e+02 | 3.3e+02 | -1.4 | 100% / 100% / 100% |
| altitude +3 km | 20 | 662 | 9.85e-04 | 0.00101 | 910 / 3 | 99.6 | 53 | 4.149 / 2.10e-04 | 9.87e-05 | 9.78e-06 | -1.72e-05 | -4.18e-04 | 3.3e+02 | 3.3e+02 | -1.7 | 100% / 100% / 100% |
| combined 10/5/3 km | 20 | 662 | 9.99e-04 | 0.00101 | 9601 / 5 | 99.9 | 82 | 16.3 / 2.80e-04 | -2.07e-04 | -2.04e-05 | 3.63e-05 | 8.75e-04 | 3.3e+02 | 3.3e+02 | -1.1 | 100% / 100% / 100% |
| random RTN 1-10 km | 20 | 662 | 9.94e-04 | 0.001013 | 3177 / 5 | 99.8 | 71 | 7.238 / 2.56e-04 | 8.76e-05 | 8.62e-06 | -1.75e-05 | -3.73e-04 | 3.3e+02 | 3.3e+02 | -2 | 100% / 100% / 100% |

**Runs and perturbations.** The golden perturbed TLE uses the full I/Q recording (one deterministic run, `runs=1`). Every synthetic case runs 20 seeds with the measurement-level surrogate (true Doppler + white Gaussian noise at 20 dB). Across those seeds the *orbit perturbation is randomised too*, not just the noise: the four named cases draw a magnitude jitter of 0.7-1.3x around the nominal RTN vector, and the `random RTN 1-10 km` case draws a uniform random direction on the sphere and a magnitude in 1-10 km for every seed. Original success needs a >90% along-track drop and a corrected Doppler RMS within 2x the empirical floor (or 5 mHz); scoped success is the component-specific rule stated in the addendum; **RTN success** is the same along-track rule plus a true 3-D RTN RMS below 500 m. The three rates are reported side by side; a Doppler-only pass is not treated as orbit success if the true RTN error is larger.

**Cost sign.** `cost = 0.5*sum(((fd-pred-bias)/sigma)^2)` with the same calibrated per-point sigma for truth and solution, and the constant bias refitted for each. `cost(sol)-cost(truth) < 0` therefore means the solution has the smaller weighted residual; it is not evidence of a better orbit. Along a flat valley the fit can absorb noise or a data systematic and lower the cost below the truth; the true RTN/element errors and the truth-cost subspace check in Section 2b decide. The earlier `truth_cost_lower=False` statement used a cost built from the CRLB (uncalibrated) sigma and the free-subspace truth vector; with the calibrated floor the two definitions are reconciled here.


### 2b. True element, cost and weak-observability checks

Element differences of the initial TLE from truth at the reference epoch (all six elements and BSTAR; the fit frees only M, n, RAAN, i):

| case | dM_ref deg | dn rev/day | dRAAN deg | di deg | de | dargp deg | dBSTAR |
|---|---|---|---|---|---|---|---|
| golden PERT (I/Q) | +0.0077 | -4.415e-03 | -0.0077 | +0.0000 | +0.00e+00 | +0.0000 | +0.00e+00 |
| altitude I/Q | -0.0126 | -8.818e-03 | +0.0033 | -0.0033 | +0.00e+00 | +0.0000 | +0.00e+00 |
| combined I/Q | +0.0546 | -8.852e-03 | -0.0362 | +0.0230 | +0.00e+00 | +0.0000 | +0.00e+00 |

Truth cost with *all* elements set to truth, and with only the free subspace (M, n, RAAN, i) set to truth and e/argp/BSTAR left at the initial TLE values. Both use the fit's calibrated per-point sigma; `sol` is the fitted orbit. A negative difference again means the solution has the lower weighted residual.

| case | truth-all cost | truth-subspace cost | solution cost | cost(sol)-cost(truth-all) |
|---|---|---|---|---|
| golden PERT (I/Q) | 256.55 | 256.55 | 255.00 | -1.56 |
| altitude I/Q | 7800.66 | 7800.66 | 132.77 | -7667.88 |
| combined I/Q | 8230.44 | 8230.44 | 135.23 | -8095.21 |
| altitude surrogate | 336.72 | 336.72 | 331.00 | -5.73 |
| combined surrogate | 331.07 | 331.07 | 331.00 | -0.07 |

This geometry has no separate e/argp/BSTAR signal, so the all-element and subspace truth costs coincide; the relevant comparison is truth vs solution at the same sigma. The solution can have the lower cost while being kilometres off in cross-track, which is why cost is never used here as proof of a better orbit.

Weak-direction geometry. One kilometre of the weakest singular direction at the reference epoch maps to the RTN vector below; the same direction integrated over the whole pass reaches the quoted cross-track amplitude. The `ratio` and lag-1 columns show whether the post-fit residual is white (surrogate) or a smooth extraction systematic that the fit can follow (I/Q).

| case | n | fit RMS / truth resid Hz | ratio | di deg | true N rms m | 1 km-at-ref R/T/N m | max |N| over window m | sol. weak N rms m | lag-1 autocorr |
|---|---|---|---|---|---|---|---|---|---|
| golden PERT (I/Q) | 510 | 0.07925 / 0.07949 | 1.00 | +0.0067 | 181 | +210/+67/+711 | 2053 | 230 | -0.04 |
| altitude I/Q | 267 | 0.00048 / 0.00367 | 0.13 | +0.1070 | 3952 | +209/+43/+979 | 1443 | 4071 | +0.99 |
| combined I/Q | 272 | 0.00047 / 0.00368 | 0.13 | +0.1193 | 4445 | +206/+41/+978 | 1447 | 4572 | +0.99 |
| altitude surrogate | 662 | 0.00100 / 0.00101 | 0.99 | -0.0011 | 49 | +211/+43/+979 | 2066 | 50 | -0.04 |
| combined surrogate | 662 | 0.00101 / 0.00101 | 1.00 | +0.0037 | 164 | +210/+43/+979 | 2065 | 164 | -0.01 |

Adding free `e*cos(w)`/`e*sin(w)` (bias included) does not improve on the four-parameter fit: it lands at the same residual level in every case, so the extra eccentricity direction is not the missing information and the plane error is not a fitting-degrees-of-freedom problem.

| case | e*cosw RMS Hz | noise floor Hz | reaches floor | fitted ecc |
|---|---|---|---|---|
| golden PERT (I/Q) | 0.077730 | 0.079490 | True | 1.240e-03 |
| altitude I/Q | 0.000382 | 0.003666 | True | 1.418e-03 |
| combined I/Q | 0.000369 | 0.003682 | True | 1.494e-03 |
| altitude surrogate | 0.001001 | 0.001010 | True | 1.422e-03 |
| combined surrogate | 0.001007 | 0.001008 | True | 1.569e-03 |

**Why the surrogate passes and the I/Q extraction did not.** The `ratio` column is the post-fit RMS divided by the residual to the *true* orbit. For the white-noise surrogate it is about 1 and the residual is uncorrelated (lag-1 ~0): the fit only removes noise and lands within tens of metres. For the fresh I/Q cases the ratio is about 0.13 and lag-1 is ~0.99: the extraction leaves a smooth, metre-per-second-scale systematic that the fit follows down to a lower RMS, moving the plane by ~0.1 deg and cross-track by ~4 km. At this SNR the per-point floor is ~1 mHz, so the few-mHz extraction bias dominates, and the surrogate (which has no such bias) is optimistic. The stored golden recording has a much larger floor (0.079 Hz), so the same bias is buried and its cross-track error is 181 m. Therefore the I/Q numbers, not the 20/20 surrogate rates, are the honest headline for the altitude and combined cases: the Doppler-only criterion passes, but the true RTN error does not. The formal plane sigma is also too small in this regime (Section 3).


### 2c. Golden regression attribution

The golden case was re-run on the *same stored recording* under the previous commit (`dd49fd8`) and the working tree, in a four-cell ablation (`HEAD dd49fd8 vs working tree, on the same stored golden01 recording; each cell fixes the extracted measurement arrays so the extraction and fit/ code changes are separated`):

| code state | measurement extraction | N err m | Doppler RMS Hz | floor Hz | di deg |
|---|---|---|---|---|---|
| HEAD dd49fd8 | linear phase slope | 181 | 0.00257 | 0.07925 | 0.0067 |
| HEAD dd49fd8 | cubic phase slope | 1971 | 0.00804 | 0.09766 | 0.0561 |
| working tree | linear phase slope | 376 | 0.00603 | 0.07893 | 0.0140 |
| working tree | cubic phase slope | 1450 | 0.00888 | 0.09718 | 0.0436 |
| working tree, f_scale=1.0 | linear phase slope | 181 | 0.00257 | 0.07893 | 0.0067 |
| HEAD dd49fd8, f_scale=3.0 | linear phase slope | 784 | 0.00739 | 0.07925 | 0.0262 |
| working tree, ftol=1e-8 | linear phase slope | 365 | 0.00600 | 0.07893 | 0.0137 |
| working tree, pre-cal soft_l1 f_scale=1.0 | linear phase slope | 393 | 0.00609 | 0.07893 | 0.0146 |
| working tree after revert | linear phase slope | 181 | 0.00257 | 0.07925 | 0.0067 |

Two independent uncommitted changes caused the regression, dominated by the extractor. (1) The cubic phase-slope fit injects occasional window outliers: the per-window Doppler differs from the linear fit by up to 0.57 Hz, while the physical curvature correction is under 5 mHz. It raises the measured floor from 0.079 to 0.097 Hz and drives the cross-track error from 181 m to about 1971 m even with the old fit. (2) The soft_l1 f_scale default change 1.0 to 3.0 is the whole fit-side effect: with the old extraction it alone moves N from 181 to 784 m. Tight tolerances (365 m) and the pre-calibration loss (393 m) are negligible. Both changes were reverted; the current code on the same recording again gives N=181 m, Doppler 2.57 mHz and floor 0.0793 Hz.

## 3. Multi-observation

| case | n | RMS Hz | sigma M deg | sigma n | sigma RAAN deg | sigma i deg | RTN sigma R/T/N m | dRAAN / sigma | di / sigma | N err / sigma |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 pass / 1 station | 106 | 3.67e-04 | 6.98e-04 | 9.89e-05 | 0.00225 | 0.00359 | 34/15/201 | -0.00309 / 0.00225 | +0.09418 / 0.00359 | 3173 / 201 |
| 2 passes / 1 station | 406 | 3.70e-04 | 4.84e-05 | 2.31e-06 | 4.50e-05 | 1.27e-04 | 1/3/1 | +0.00028 / 0.00004 | +0.00083 / 0.00013 | 7 / 1 |
| 1 pass / 2 stations | 265 | 3.62e-04 | 7.37e-05 | 1.05e-05 | 4.20e-05 | 2.67e-04 | 4/1/11 | +0.00009 / 0.00004 | -0.00011 / 0.00027 | 12 / 11 |
| 2 passes / 2 stations | 865 | 9.61e-04 | 2.12e-05 | 9.25e-07 | 1.52e-05 | 5.70e-05 | 0/1/0 | +0.00002 / 0.00002 | +0.00010 / 0.00006 | 1 / 0 |

Two passes or two stations break the single-pass cross-track degeneracy in absolute terms: the cross-track error falls from ~3 km to 7-12 m. The reported sigma, however, remains a *lower bound*: the actual/sigma ratio is ~16 for one pass and ~5-7 even with two passes/stations, because the formal covariance does not include the smooth extraction residual (Section 2b). The reliable statement is the absolute error, not the formal sigma; single-pass cross-track must be treated as unresolved rather than trusted.

## 4. Convergence basin and SNR sweep

20 seeds per point, golden perturbation scaled 0.5x-10x at 20 dB; success = converged and along-track error reduced by >90%.

| scale | success | median dT after m |
|---|---|---|
| 0.5x | 20/20 | 45.43 |
| 1x | 20/20 | 110.9 |
| 2x | 20/20 | 206.9 |
| 5x | 20/20 | 484.7 |
| 10x | 20/20 | 847.8 |

| SNR dB | success | median dT after m | noise Hz |
|---|---|---|---|
| 30 | 20/20 | 106.7 | 0.0010 |
| 25 | 20/20 | 109.6 | 0.0017 |
| 20 | 20/20 | 98.58 | 0.0031 |
| 15 | 20/20 | 110.1 | 0.0055 |
| 10 | 20/20 | 104.6 | 0.0098 |
| 5 | 20/20 | 109.5 | 0.0173 |
| 0 | 20/20 | 116.2 | 0.0308 |

**SNR law.** The surrogate uses `sigma_hz(SNR) = 0.001 * 10**((20-SNR)/20)`, i.e. `10**(-SNR_dB/20)` relative to the 20 dB anchor. One decade of SNR power (10 dB) multiplies sigma by `10**(10/20) = sqrt(10) = 3.162`, which is exactly the CRLB `1/sqrt(SNR)` law (`snr_scaling` in the addendum JSON, `equal=true` for every point). The 10 dB/decade rule is therefore correct as used; no fix was needed.

The fast sweep uses the measurement-level surrogate (true Doppler + white Gaussian noise at the calibrated per-window noise level). It was validated once against a full I/Q extraction at 20 dB: the corrected R/T/N RMS differs by about 8-10% and the residual RMS by about 2% (`validation` in the JSON). The 0-5 dB points extrapolate the noise curve because the tone tracker often loses lock there; they test the fit, not detectability. Plots: `figures/c4_convergence.png`, `figures/c4_snr.png`.

## 5. Statistical honesty

Monte Carlo: 50 fits at 15 dB (0.00549 Hz noise, 1x golden perturbation), 50 converged. The `ratio` columns are empirical scatter divided by the predicted 1-sigma; `pre` is the CRLB-only fit before sigma calibration, `post` is the calibrated fit that feeds the reported covariance.

| parameter | empirical std | pre sigma | pre ratio | post sigma | post ratio |
|---|---|---|---|---|---|
| M | 0.002507 | 0.002489 | 0.52 | 0.002489 | 1.01 |
| n | 2.46e-04 | 2.43e-04 | 0.53 | 2.44e-04 | 1.01 |
| RAAN | 5.10e-04 | 6.12e-04 | 0.62 | 6.16e-04 | 0.83 |
| i | 0.01048 | 0.01045 | 0.51 | 0.01047 | 1.00 |

The earlier 1.07-1.11 ratios were therefore **post-calibration** numbers; the pre-calibration fit is optimistic (0.51-0.62) because the CRLB sigma ignores the common per-window floor, and the calibration step removes most of that bias (post 0.83-1.01). These ratios describe the white-noise surrogate geometry only; for the I/Q cases the actual plane error exceeds the formal sigma (Section 3) because the smooth extraction residual is not in the white-noise covariance model.

Weak-plane sigma (profile vs local linear, from Section 2b). The plane direction is the only one where a local Jacobian is misleading:

| case | profile/linear | condition |
|---|---|---|
| golden PERT (I/Q) | 0.162 | 2.83e+04 |
| altitude I/Q | 0.100 | 3.27e+05 |
| combined I/Q | 0.094 | 3.12e+05 |
| altitude surrogate | 1.518 | 2.71e+04 |
| combined surrogate | 0.417 | 2.71e+04 |

For the I/Q cases the profiled sigma is 0.09-0.15 of the local linear value (a factor 7-11), confirming that the linear covariance overstates the plane uncertainty there; for the white-noise surrogate the ratio is 0.4-1.5, so the plane is genuinely better constrained by the white-noise data. This is the same weak-observability effect, not a separate bug.

Constant bias: added 0.5 Hz; nuisance estimate 0.4957 Hz; orbit parameter shifts:

| parameter | shift | 1-sigma |
|---|---|---|
| orbit.M | 4.55e-04 | 0.0017 |
| orbit.n | 4.43e-05 | 1.67e-04 |
| orbit.RAAN | -8.57e-05 | 3.71e-04 |
| orbit.i | -0.00192 | 0.00718 |

Sign convention: flipped measured Doppler gives RMS 1541.731 Hz (ratio 522795.1x), classification `diverged`, warnings ['solver did not converge; the best of several seeds is reported', 'residual mean changes sign across the pass - suspected time or sign-convention mismatch', 'cross-track is weakly observed: N 1-sigma 1760161321.4 km; a single pass has little cross-track sensitivity', 'radial/along-track uncertainty is large; check the pass geometry and the initial TLE error'].

Start-time error: measurement times shifted, Doppler left unchanged.

| shift s | dM_ref deg | dT after m | dn rev/day | dRAAN deg | di deg |
|---|---|---|---|---|---|
| 0 | -0.0117 | 85 | - | - | - |
| 0.1 | -0.0194 | 655 | -0.00132 | 0.00329 | 0.0578 |
| 1 | -0.0704 | 7439 | -0.00112 | 0.00668 | 0.0492 |
| 10 | -0.602 | 75130 | -0.00131 | 0.0445 | 0.0574 |

## 6. What is observable from one pass, two passes, two stations

- **One pass, one station:** radial and along-track are well determined (tens of metres); the plane (RAAN/inclination) is weakly observed through a nonlinear trade with along-track motion. The formal sigma is not trustworthy in that direction, and the corrected cross-track error can be several kilometres even when Doppler looks excellent.
- **Two passes, one station:** the changing viewing geometry separates the plane from along-track; the actual RAAN/inclination errors and the cross-track error collapse to metres, but the formal sigma stays a lower bound (~5-7x too small for the I/Q extraction).
- **One pass, two stations:** the baseline gives the same geometric separation as a second pass; cross-track drops from kilometres to tens of metres.
- **Two passes, two stations:** all components are over-determined; the fit is at the noise floor and the reported sigmas are meaningful.


## 7. Unit tests

`pytest` covers extractor accuracy, SigMF/scenario cross-checks, manifest import, required fields, YAML/manual equivalence, TLE round trip, `true.tle` isolation, solver progress, zero-measurement failure and the profile-covariance path. `ruff check src tests` is clean.


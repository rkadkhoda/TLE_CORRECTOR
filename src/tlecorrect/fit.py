"""Nonlinear least squares of SGP4 elements against Doppler measurements.

Parameterisation
----------------
The orbital unknowns are increments relative to the input TLE, re-centred at
``t_ref`` (the middle of the analysis window):

* ``M``: mean anomaly increment at ``t_ref`` (deg),
* ``n``: mean-motion increment (rev/day),
* ``RAAN``: increment (deg),
* ``i``: increment (deg).

``n`` and ``M`` are re-centred together so that their strong correlation is
removed: the reference mean anomaly is ``M_ref = M_epoch + n*360*(t_ref -
epoch)``; changing ``M_ref`` shifts the pass time, changing ``n`` changes the
orbital period.  The mapping back to the TLE epoch uses exactly the same
expressions, so no information is lost before TLE rounding.

``e``, ``argp``, ``BSTAR`` and the epoch stay fixed at the input TLE values.
Each observation carries its own constant frequency bias (and optional linear
drift), estimated by default.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .frames import rtn_basis
from .measurement import DopplerModel, MeasurementSet
from .sgp4wrap import OrbitElements, OrbitModel
from .timeutil import jd_split_many

__all__ = ["FitResult", "ParamSpace", "FitEngine", "run_fit"]

ORBIT_NAMES = ("M", "n", "RAAN", "i")
_DEFAULT_SCALE = {"M": 1.0, "n": 1e-4, "RAAN": 1.0, "i": 1e-3}


@dataclass
class ParamSpace:
    """Map the optimizer state vector to orbit elements and nuisances."""

    orbit_model: OrbitModel
    t_ref: dt.datetime
    free: tuple[str, ...] = ORBIT_NAMES
    bias_mode: str = "estimate"
    drift: bool = False
    observation_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.base = self.orbit_model.base_elements
        self.dt_ref_days = (
            self.t_ref - self.orbit_model.base_tle.epoch
        ).total_seconds() / 86400.0
        self.m_ref0 = self.base.M + self.base.n * 360.0 * self.dt_ref_days

    # -- layout ------------------------------------------------------------
    @property
    def n_orbit(self) -> int:
        return len(self.free)

    @property
    def n_bias(self) -> int:
        return len(self.observation_ids) if self.bias_mode == "estimate" else 0

    @property
    def n_drift(self) -> int:
        return len(self.observation_ids) if self.drift else 0

    @property
    def size(self) -> int:
        return self.n_orbit + self.n_bias + self.n_drift

    def names(self) -> list[str]:
        names = [f"orbit.{k}" for k in self.free]
        if self.bias_mode == "estimate":
            names += [f"obs{oid}.bias_hz" for oid in self.observation_ids]
        if self.drift:
            names += [f"obs{oid}.drift_hz_s" for oid in self.observation_ids]
        return names

    def x_scale(self, override: list[float] | None = None) -> np.ndarray:
        if override is not None:
            if len(override) != self.size:
                raise ValueError(f"x_scale has {len(override)} entries, expected {self.size}")
            return np.asarray(override, dtype=float)
        scale = [_DEFAULT_SCALE[k] for k in self.free]
        scale += [100.0] * self.n_bias
        scale += [1.0] * self.n_drift
        return np.asarray(scale, dtype=float)

    def lower_bounds(self) -> np.ndarray:
        lim = {"M": (-360.0, 360.0), "n": (-0.5, 0.5), "RAAN": (-180.0, 180.0),
               "i": (-45.0, 45.0)}
        lo = [lim[k][0] for k in self.free]
        lo += [-1e6] * self.n_bias
        lo += [-1e4] * self.n_drift
        return np.asarray(lo, dtype=float)

    def upper_bounds(self) -> np.ndarray:
        lim = {"M": (-360.0, 360.0), "n": (-0.5, 0.5), "RAAN": (-180.0, 180.0),
               "i": (-45.0, 45.0)}
        hi = [lim[k][1] for k in self.free]
        hi += [1e6] * self.n_bias
        hi += [1e4] * self.n_drift
        return np.asarray(hi, dtype=float)

    # -- mapping -----------------------------------------------------------
    def elements(self, theta: np.ndarray) -> OrbitElements:
        inc = {k: 0.0 for k in ORBIT_NAMES}
        for k, v in zip(self.free, theta[: self.n_orbit], strict=True):
            inc[k] = float(v)
        n = self.base.n + inc["n"]
        m_ref = self.m_ref0 + inc["M"]
        m_epoch = m_ref - n * 360.0 * self.dt_ref_days
        return OrbitElements(
            n=n,
            ecc=self.base.ecc,
            i=self.base.i + inc["i"],
            raan=(self.base.raan + inc["RAAN"]) % 360.0,
            argp=self.base.argp,
            M=m_epoch % 360.0,
        )

    def nuisances(self, theta: np.ndarray) -> dict:
        out: dict[int, dict[str, float]] = {}
        pos = self.n_orbit
        for oid in self.observation_ids:
            nz = out.setdefault(oid, {"bias_hz": 0.0, "drift_hz_s": 0.0})
            if self.bias_mode == "estimate":
                nz["bias_hz"] = float(theta[pos])
                pos += 1
        for oid in self.observation_ids:
            if self.drift:
                out.setdefault(oid, {"bias_hz": 0.0, "drift_hz_s": 0.0})[
                    "drift_hz_s"
                ] = float(theta[pos])
                pos += 1
        return out

    def pack(self, orbit_inc: dict[str, float], biases: dict[int, float],
             drifts: dict[int, float] | None = None) -> np.ndarray:
        theta = np.zeros(self.size)
        for i, k in enumerate(self.free):
            theta[i] = orbit_inc.get(k, 0.0)
        pos = self.n_orbit
        if self.bias_mode == "estimate":
            for oid in self.observation_ids:
                theta[pos] = biases.get(oid, 0.0)
                pos += 1
        if self.drift:
            for oid in self.observation_ids:
                theta[pos] = (drifts or {}).get(oid, 0.0)
                pos += 1
        return theta


@dataclass
class FitResult:
    elements: OrbitElements
    theta: np.ndarray
    param_names: list[str]
    covariance: np.ndarray
    correlation: np.ndarray
    sigma: np.ndarray
    residual_hz: np.ndarray
    residual_weighted: np.ndarray
    pred_hz: dict[int, np.ndarray]
    measurements: list[MeasurementSet]
    cost: float
    reduced_chi2: float
    rms_hz: float
    rms_hz_per_obs: dict[int, float]
    success: bool
    status: int
    message: str
    nfev: int
    njev: int
    iteration_log: list[dict] = field(default_factory=list)
    classification: str = ""
    warnings: list[str] = field(default_factory=list)
    observability: dict = field(default_factory=dict)
    rtn_sigma: dict = field(default_factory=dict)
    initial_cost: float = float("nan")
    tca_shift_s: float = 0.0
    sigma_scale: dict = field(default_factory=dict)
    sigma_floor_hz: dict = field(default_factory=dict)
    sigma_inflation: float = 1.0
    precalibration: dict = field(default_factory=dict)

    @property
    def param_dict(self) -> dict[str, float]:
        return dict(zip(self.param_names, self.theta, strict=True))


class FitEngine:
    def __init__(self, orbit_model: OrbitModel, space: ParamSpace,
                 model: DopplerModel, stations: dict,
                 measurements: list[MeasurementSet],
                 loss: str = "soft_l1", f_scale: float = 1.0,
                 max_nfev: int = 200, tikhonov: bool = False,
                 prior_sigma: dict[str, float] | None = None,
                 tikhonov_target: OrbitElements | None = None,
                 ftol: float = 1e-12, xtol: float = 1e-12, gtol: float = 1e-12,
                 profile_sigma: bool = True):
        self.orbit_model = orbit_model
        self.space = space
        self.model = model
        self.stations = stations
        self.measurements = measurements
        self.loss = loss
        self.f_scale = float(f_scale)
        self.max_nfev = int(max_nfev)
        self.tikhonov = bool(tikhonov)
        self.prior_sigma = prior_sigma or {}
        self.tikhonov_target = tikhonov_target or orbit_model.base_elements
        self.ftol = float(ftol)
        self.xtol = float(xtol)
        self.gtol = float(gtol)
        self.profile_sigma = bool(profile_sigma)
        self._prior_cache: np.ndarray | None = None

    # -- residuals ---------------------------------------------------------
    def _raw(self, theta: np.ndarray):
        el = self.space.elements(theta)
        nuis = self.space.nuisances(theta)
        r, pred = self.model.residuals(el, self.stations, self.measurements, nuis)
        return r, pred, el

    def _residual(self, theta: np.ndarray) -> np.ndarray:
        r, _, _ = self._raw(theta)
        if not self.tikhonov or not self.prior_sigma:
            return r
        prior = []
        target_el = self.tikhonov_target
        cur_el = self.space.elements(theta)
        target_map = {
            "M": target_el.M + target_el.n * 360.0 * self.space.dt_ref_days,
            "n": target_el.n,
            "RAAN": target_el.raan,
            "i": target_el.i,
        }
        cur_map = {
            "M": self.space.m_ref0 + (cur_el.M % 360.0 - self.space.base.M
                                      - self.space.base.n * 360.0 * self.space.dt_ref_days),
            "n": cur_el.n,
            "RAAN": cur_el.raan,
            "i": cur_el.i,
        }
        for k in self.space.free:
            ps = self.prior_sigma.get(k)
            if ps and ps > 0:
                prior.append((cur_map[k] - target_map[k]) / ps)
        if not prior:
            return r
        self._prior_cache = np.asarray(prior)
        return np.concatenate([r, self._prior_cache])

    # -- initialisation ----------------------------------------------------
    def initial_shift(self, max_shift_s: float = 120.0, step_s: float = 5.0) -> tuple[float, np.ndarray]:
        """Grid-search an along-track (mean-anomaly) shift that aligns the pass.

        The measured time of closest approach is not used directly because an
        unknown frequency bias moves the zero crossing; the steepest-slope
        point is bias-independent.  A coarse grid around the initial TLE is
        also the multi-start check that exposes local minima.
        """
        el0 = self.orbit_model.base_elements
        ms0 = self.measurements[0]
        t = ms0.t_rel_s
        fd = ms0.fd_hz
        if len(t) >= 8:
            k = int(np.argmax(np.abs(np.gradient(fd, t))))
            t_meas = float(t[k])
            jd, fr = jd_split_many(ms0.times_utc)
            fd_model = self.model.orbit.doppler(el0, self.stations[ms0.station_name],
                                                jd, fr, self.model.fc_hz)
            k2 = int(np.argmax(np.abs(np.gradient(fd_model, t))))
            t_model = float(t[k2])
            guess_s = t_meas - t_model
        else:
            guess_s = 0.0
        candidates = list(np.arange(-max_shift_s, max_shift_s + step_s / 2, step_s))
        candidates.append(float(np.clip(guess_s, -max_shift_s, max_shift_s)))
        best = (float("inf"), 0.0)
        for shift in candidates:
            dM = -360.0 * el0.n * shift / 86400.0
            inc = {"M": dM, "n": 0.0, "RAAN": 0.0, "i": 0.0}
            theta = self.space.pack(inc, {oid: 0.0 for oid in self.space.observation_ids})
            r, _, _ = self._raw(theta)
            cost = self._robust_cost(r)
            if cost < best[0]:
                best = (cost, float(shift))
        return best[1], self.space.pack(
            {"M": -360.0 * el0.n * best[1] / 86400.0},
            {oid: 0.0 for oid in self.space.observation_ids},
        )

    def initial_biases(self, theta: np.ndarray) -> np.ndarray:
        theta = theta.copy()
        r, pred, _ = self._raw(theta)
        pos = self.space.n_orbit
        if self.space.bias_mode != "estimate":
            return theta
        for oid in self.space.observation_ids:
            ms = next(m for m in self.measurements if m.observation_id == oid)
            p = pred[oid]
            sigma = np.maximum(ms.sigma_hz, 1e-12)
            # weighted median of the raw residuals is robust to outliers
            raw = ms.fd_hz - p
            order = np.argsort(raw)
            cum = np.cumsum(sigma[order] ** -2)
            idx = int(np.searchsorted(cum, cum[-1] / 2.0))
            theta[pos] = float(raw[order][min(idx, len(raw) - 1)])
            pos += 1
        return theta

    def _robust_cost(self, r: np.ndarray) -> float:
        if self.loss == "soft_l1":
            z = np.sqrt(1.0 + (r / self.f_scale) ** 2)
            return float(np.sum(2.0 * (z - 1.0)))
        if self.loss == "huber":
            a = np.abs(r / self.f_scale)
            return float(np.sum(np.where(a <= 1.0, a**2, 2.0 * a - 1.0)))
        return float(np.sum(r**2))

    # -- solve -------------------------------------------------------------
    def solve(self, max_shift_s: float = 120.0,
              theta_start: np.ndarray | None = None) -> FitResult:
        if theta_start is None:
            shift, theta0 = self.initial_shift(max_shift_s=max_shift_s)
            theta0 = self.initial_biases(theta0)
        else:
            shift, theta0 = 0.0, np.asarray(theta_start, dtype=float).copy()
        lo, hi = self.space.lower_bounds(), self.space.upper_bounds()
        theta0 = np.clip(theta0, lo, hi)
        r_init, _, _ = self._raw(theta0)
        initial_cost = float(np.sqrt(np.mean((r_init * np.concatenate(
            [np.maximum(m.sigma_hz, 1e-12) for m in self.measurements])) ** 2)))
        log: list[dict] = []

        def callback(xk):
            r, _, _ = self._raw(np.asarray(xk))
            log.append({
                "iteration": len(log) + 1,
                "cost": float(0.5 * np.sum(r**2)),
                "gradient_norm": float("nan"),
                "step_norm": float("nan"),
            })

        best = None
        seeds = [theta0]
        # Deterministic retry seeds: never let the solver stop at iteration 0
        # with a large residual.
        for dm in (0.5, -0.5, 2.0, -2.0):
            s = theta0.copy()
            if "M" in self.space.free:
                s[self.space.free.index("M")] += dm
            seeds.append(s)
        def run_seed(seed):
            nonlocal best
            log.clear()
            res = least_squares(
                self._residual, seed, bounds=(lo, hi), method="trf",
                x_scale=self.space.x_scale(), loss=self.loss, f_scale=self.f_scale,
                max_nfev=self.max_nfev, callback=callback,
                ftol=self.ftol, xtol=self.xtol, gtol=self.gtol,
            )
            r, pred, el = self._raw(res.x)
            raw_rms = float(np.sqrt(np.mean((r * np.concatenate(
                [np.maximum(m.sigma_hz, 1e-12) for m in self.measurements])) ** 2)))
            cand = (raw_rms, res, r, pred, el, list(log), 0)
            if best is None or raw_rms < best[0]:
                best = cand
            return res

        for seed in seeds:
            res = run_seed(seed)
            if res.success and seed is seeds[0] and res.nfev > 2:
                break

        score, res, r, pred, el, log, seed_idx = best
        return self._finalise(res, r, pred, el, log, shift, score, len(seeds) > 1,
                              initial_cost)

    def _numerical_jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Central-difference Jacobian of the weighted residual at ``theta``.

        scipy's ``res.jac`` is not used for the covariance: with ``x_scale``
        and robust losses its column scaling is not the physical one.  Explicit
        central differences with per-parameter steps keep the covariance and
        the condition number in physical units.
        """
        base = self._residual(theta)
        J = np.zeros((len(base), self.space.size))
        steps = []
        for name in self.space.names():
            if name == "orbit.n":
                steps.append(1e-6)
            elif name.endswith("drift_hz_s"):
                steps.append(1e-5)
            elif name.endswith("bias_hz"):
                steps.append(1e-4)
            else:
                steps.append(1e-4)
        for j, h in enumerate(steps):
            xp = theta.copy()
            xm = theta.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (self._residual(xp) - self._residual(xm)) / (2.0 * h)
        return J

    def _finalise(self, res, r, pred, el, log, shift, raw_rms, multi,
                  initial_cost=float("nan")) -> FitResult:
        sigma_vec = np.concatenate([np.maximum(m.sigma_hz, 1e-12) for m in self.measurements])
        dof = max(1, len(r) - self.space.size)
        redchi = float(np.sum(r**2) / dof)
        try:
            J = self._numerical_jacobian(np.asarray(res.x, dtype=float))
        except (ValueError, FloatingPointError):
            J = np.asarray(res.jac, dtype=float)
        inflation, rho = self._inflation_factor(r)
        try:
            U, sv, Vt = np.linalg.svd(J, full_matrices=False)
            target = redchi * inflation**2
            amps = []
            for k in range(len(sv)):
                linear = 1.0 / sv[k] if sv[k] > 0 else 0.0
                if self.profile_sigma and k == len(sv) - 1:
                    # The weakest direction is a nonlinear plane/along-track
                    # cancellation.  Its local Jacobian is nearly singular
                    # and would overstate the uncertainty, so it is profiled
                    # out to Delta chi^2 = 1.  Well-conditioned directions
                    # use the linear curvature.
                    amps.append(self._profile_amplitude(
                        np.asarray(res.x, dtype=float), Vt[k],
                        float(np.sum(r**2)), target, linear))
                else:
                    amps.append(float(np.sqrt(max(target, 0.0))) * linear)
            cov = (Vt.T * np.asarray(amps) ** 2) @ Vt
            cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            cov = np.full((self.space.size, self.space.size), np.nan)
            sv = np.array([np.nan])
            cond = float("inf")
        diag = np.diag(cov)
        sig = np.sqrt(np.maximum(diag, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(sig, sig)
        corr = np.nan_to_num(corr, nan=0.0)

        resid_hz = r * sigma_vec
        rms_per = {}
        for ms, sl in zip(self.measurements, self._slices(), strict=True):
            rr = resid_hz[sl]
            rms_per[ms.observation_id] = float(np.sqrt(np.mean(rr**2))) if rr.size else float("nan")

        warnings: list[str] = []
        if cond > 1e10:
            warnings.append(
                f"Jacobian is ill-conditioned (condition number {cond:.2e}); "
                "some parameters are poorly constrained by this data set."
            )
        if multi and not res.success:
            warnings.append("solver did not converge; the best of several seeds is reported")
        # Time/convention mismatch signature: a *large* residual mean that
        # changes sign halfway.  A small sign change is normal noise.
        rms_hz = float(np.sqrt(np.mean(resid_hz**2))) if len(resid_hz) else 0.0
        if len(resid_hz) > 8 and rms_hz > 0:
            half = len(resid_hz) // 2
            m1, m2 = float(np.mean(resid_hz[:half])), float(np.mean(resid_hz[half:]))
            if m1 * m2 < 0 and max(abs(m1), abs(m2)) > 0.25 * rms_hz:
                warnings.append(
                    "residual mean changes sign across the pass - suspected time or "
                    "sign-convention mismatch"
                )

        obs = self._observability(cov, cond, sv)
        rtn = self._rtn_uncertainty(cov, res.x, el)
        if rtn:
            sig_m = rtn["sigma_m"]
            if sig_m["N"] > 1_000.0:
                warnings.append(
                    f"cross-track is weakly observed: N 1-sigma {sig_m['N'] / 1000.0:.1f} km; "
                    "a single pass has little cross-track sensitivity"
                )
            if sig_m["T"] > 10_000.0 or sig_m["R"] > 10_000.0:
                warnings.append(
                    "radial/along-track uncertainty is large; check the pass geometry "
                    "and the initial TLE error"
                )
        classification = self._classify(res, cond, len(r), self.space.size)
        return FitResult(
            elements=el, theta=np.asarray(res.x), param_names=self.space.names(),
            covariance=cov, correlation=corr, sigma=sig, residual_hz=resid_hz,
            residual_weighted=r, pred_hz=pred, measurements=self.measurements,
            cost=float(0.5 * np.sum(r**2)), reduced_chi2=redchi,
            rms_hz=float(np.sqrt(np.mean(resid_hz**2))), rms_hz_per_obs=rms_per,
            success=bool(res.success), status=int(res.status), message=str(res.message),
            nfev=int(res.nfev), njev=int(res.njev), iteration_log=log,
            classification=classification, warnings=warnings,
            observability=obs, rtn_sigma=rtn, initial_cost=initial_cost,
            tca_shift_s=float(shift), sigma_inflation=float(inflation),
            precalibration={},
        )

    def _profile_amplitude(self, theta: np.ndarray, direction: np.ndarray,
                           chi0: float, target: float,
                           linear: float) -> float:
        """Find the step along ``direction`` that raises chi^2 by ``target``.

        The linear (Cramer-Rao) answer is ``sqrt(target)/s``; for a direction
        whose cost is visibly curved already at the 1-sigma scale, the actual
        profile is smaller.  This keeps the reported sigma honest for the
        weak plane/along-track cancellation while leaving well-conditioned
        parameters at their linear value.
        """
        if not np.isfinite(linear) or linear <= 0.0:
            return 0.0
        lo, hi = self.space.lower_bounds(), self.space.upper_bounds()
        start = float(np.sqrt(max(target, 0.0)) * linear)

        def delta(a: float) -> float:
            x = theta + a * direction
            if np.any(x < lo) or np.any(x > hi):
                return float("inf")
            r = self._residual(x)
            return float(np.sum(r**2) - chi0)

        if abs(delta(start) - target) <= 1e-3 * max(target, 1.0):
            return start
        # Bracket: start from the linear scale and grow until the profile
        # crosses, but never more than a few sigma (beyond that the local
        # model is meaningless and the bounds/fallback take over).
        hi_a = max(start, 1e-12)
        for _ in range(12):
            if delta(hi_a) >= target:
                break
            hi_a *= 1.8
            if hi_a > 1.0:
                return start
        else:
            return start
        lo_a = 0.0
        for _ in range(50):
            mid = 0.5 * (lo_a + hi_a)
            if delta(mid) < target:
                lo_a = mid
            else:
                hi_a = mid
        return 0.5 * (lo_a + hi_a)

    def _inflation_factor(self, r: np.ndarray) -> tuple[float, float]:
        """Variance inflation from residual autocorrelation (one pass).

        Adjacent 1 s windows share noise and unmodelled structure; treating
        them as independent understates the parameter covariance.  The
        effective factor is ``1 + 2*sum(rho_k)`` (positive lags, truncated at
        n/3), returned as (sqrt(factor), sum(rho_k)).
        """
        r = np.asarray(r, dtype=float)
        if r.size < 8:
            return 1.0, 0.0
        r = r - r.mean()
        den = float(np.dot(r, r))
        if den <= 0:
            return 1.0, 0.0
        max_lag = min(60, r.size // 3)
        rho = 0.0
        for k in range(1, max_lag + 1):
            v = float(np.dot(r[:-k], r[k:]) / den)
            if v > 0:
                rho += v * (1.0 - k / (max_lag + 1.0))
        return float(np.sqrt(max(1.0, 1.0 + 2.0 * rho))), rho

    def _slices(self):
        out, start = [], 0
        for ms in self.measurements:
            out.append(slice(start, start + ms.n))
            start += ms.n
        return out

    def _observability(self, cov: np.ndarray, cond: float, sv: np.ndarray) -> dict:
        out = {"parameters": {}, "singular_values": [float(x) for x in np.atleast_1d(sv)],
               "condition_number": float(cond)}
        sig = np.sqrt(np.maximum(np.diag(cov), 0.0))
        for i, name in enumerate(self.space.names()):
            rec = {"sigma": float(sig[i]), "unit": _unit_for(name)}
            out["parameters"][name] = rec
        return out

    def _rtn_uncertainty(self, cov: np.ndarray, theta_sol: np.ndarray,
                         el: OrbitElements) -> dict:
        k = self.space.n_orbit
        if k == 0:
            return {}
        cov_o = cov[:k, :k]
        if not np.all(np.isfinite(cov_o)):
            return {}
        t_ref = self.space.t_ref
        jd, fr = jd_split_many([t_ref])
        r0, v0 = self.orbit_model.state_teme(el, jd[0], fr[0])
        if r0 is None:
            return {}
        basis = rtn_basis(r0, v0)
        steps = [1e-3 if name == "n" else 1e-3 for name in self.space.free]
        jac = np.zeros((3, k))
        base = np.asarray(r0)
        theta0 = np.asarray(theta_sol, dtype=float)
        for j in range(k):
            dth = theta0.copy()
            dth[j] = theta0[j] + steps[j]
            elj = self.space.elements(dth)
            rj, _ = self.orbit_model.state_teme(elj, jd[0], fr[0])
            if rj is None:
                continue
            jac[:, j] = (np.asarray(rj) - base) / steps[j]
        jac_rtn = basis @ jac
        pos_cov = jac_rtn @ cov_o @ jac_rtn.T
        sig_rtn = np.sqrt(np.maximum(np.diag(pos_cov), 0.0)) * 1000.0
        return {
            "t_ref": t_ref.isoformat(),
            "sigma_m": {"R": float(sig_rtn[0]), "T": float(sig_rtn[1]), "N": float(sig_rtn[2])},
        }

    def _classify(self, res, cond, n_data, n_par) -> str:
        if n_data < n_par:
            return "too few valid samples"
        if not np.isfinite(cond) or cond > 1e14:
            return "singular/ill-conditioned Jacobian"
        if not res.success:
            if res.status == 0:
                return "diverged"
            return f"failed: {res.message}"
        return "ok"


def _unit_for(name: str) -> str:
    if name.endswith("bias_hz"):
        return "Hz"
    if name.endswith("drift_hz_s"):
        return "Hz/s"
    return {"orbit.M": "deg", "orbit.n": "rev/day", "orbit.RAAN": "deg", "orbit.i": "deg"}.get(
        name, ""
    )


def run_fit(orbit_model: OrbitModel, measurements: list[MeasurementSet], stations: dict,
            fc_hz: float, t_ref: dt.datetime, fit_options, light_time: bool = False
            ) -> FitResult:
    """Convenience wrapper: build the parameter space and run the engine.

    When ``calibrate_sigma`` is on (the default), a first fit measures the
    empirical scatter of the residuals (RMS in Hz) per observation.  The
    per-point sigmas are then floored at that scatter before the final fit.
    The CRLB shape alone is far too optimistic for real recordings, where
    phase noise, the generator's interpolation and residual modulation add a
    common error floor; flooring keeps the fit from absorbing that floor into
    the orbit.  The floors and the implied scale factors are reported.
    """
    if not measurements or any(ms.n == 0 for ms in measurements):
        raise ValueError("no measurements to fit")
    obs_ids = tuple(ms.observation_id for ms in measurements)
    free = tuple(k for k in ORBIT_NAMES if k in (fit_options.free or []))
    space = ParamSpace(
        orbit_model=orbit_model, t_ref=t_ref, free=free,
        bias_mode=fit_options.bias_mode, drift=fit_options.drift,
        observation_ids=obs_ids,
    )
    model = DopplerModel(orbit_model, fc_hz, light_time=light_time)

    def make_engine(ms_list, loss=None, f_scale=None):
        return FitEngine(
            orbit_model, space, model, stations, ms_list,
            loss=fit_options.loss if loss is None else loss,
            f_scale=fit_options.f_scale if f_scale is None else f_scale,
            max_nfev=fit_options.max_nfev, tikhonov=fit_options.tikhonov,
            prior_sigma=fit_options.prior_sigma,
            ftol=getattr(fit_options, "ftol", 1e-12),
            xtol=getattr(fit_options, "xtol", 1e-12),
            gtol=getattr(fit_options, "gtol", 1e-12),
            profile_sigma=getattr(fit_options, "profile_sigma", True),
        )

    calibrate = (getattr(fit_options, "calibrate_sigma", True)
                 and bool(measurements))
    if calibrate:
        # The first fit must converge with the original CRLB weights.  The
        # robust loss is misleading when the residual/sigma scale is huge, so
        # start with a plain linear least-squares fit, then calibrate.
        engine = make_engine(measurements, loss="linear", f_scale=1e6)
    else:
        engine = make_engine(measurements)
    pre = engine.solve()
    current = pre
    precal = {
        "rms_hz": float(pre.rms_hz),
        "reduced_chi2": float(pre.reduced_chi2),
        "cost": float(pre.cost),
        "success": bool(pre.success),
        "nfev": int(pre.nfev),
    }
    scales = {ms.observation_id: 1.0 for ms in measurements}
    if not calibrate:
        current.sigma_scale = scales
        current.precalibration = precal
        return current
    # Two reweighting passes: the first residual RMS is itself inflated when
    # the initial elements are wrong, so the floor is recomputed once more at
    # the improved solution before the final fit.
    for _ in range(4):
        floors: dict[int, float] = {}
        pos = 0
        for ms in measurements:
            res_hz = current.residual_hz[pos:pos + ms.n]
            pos += ms.n
            sig_med = float(np.median(ms.sigma_hz)) if ms.n else 0.0
            rms = float(np.sqrt(np.mean(res_hz**2))) if ms.n else 0.0
            floors[ms.observation_id] = rms
            scales[ms.observation_id] = (rms / sig_med) if sig_med > 0 else 1.0
        scaled = []
        for ms in measurements:
            ss = np.full_like(ms.sigma_hz, floors[ms.observation_id])
            scaled.append(MeasurementSet(
                observation_id=ms.observation_id, station_name=ms.station_name,
                times_utc=ms.times_utc, t_rel_s=ms.t_rel_s, fd_hz=ms.fd_hz,
                sigma_hz=ss, quality=ms.quality,
            ))
        current = make_engine(scaled).solve(theta_start=current.theta)
        current.sigma_scale = scales
        current.sigma_floor_hz = floors
    current.precalibration = precal
    return current

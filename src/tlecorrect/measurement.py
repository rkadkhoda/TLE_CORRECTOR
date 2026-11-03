"""Measurement models: predicted Doppler and residuals.

The fit is written against the small :class:`MeasurementModel` interface so
that other measurement types (symbol/timing rate, SatNOGS waterfall or audio
adapters) can be added later without touching the optimizer.  Only the Doppler
model is implemented; the second class is a documented stub.

Conventions (from the generator documentation, confirmed against golden01):

* Doppler is positive when approaching: ``fd = -fc * dR/dt / c``.
* The range and range-rate are evaluated at the receive time; light-time
  retardation is optional and off by default (the generator does not apply it).
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .frames import C_LIGHT
from .sgp4wrap import OrbitElements, OrbitModel
from .timeutil import jd_split_many

__all__ = [
    "MeasurementSet",
    "MeasurementModel",
    "DopplerModel",
    "SymbolRateModel",
]


@dataclass
class MeasurementSet:
    """Extracted measurements for one observation, ready for the fit."""

    observation_id: int
    station_name: str
    times_utc: list[dt.datetime]
    t_rel_s: np.ndarray
    fd_hz: np.ndarray
    sigma_hz: np.ndarray
    quality: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.t_rel_s = np.asarray(self.t_rel_s, dtype=float)
        self.fd_hz = np.asarray(self.fd_hz, dtype=float)
        self.sigma_hz = np.asarray(self.sigma_hz, dtype=float)

    @property
    def n(self) -> int:
        return len(self.fd_hz)

    def t_mid_s(self) -> float:
        return float(np.median(self.t_rel_s)) if self.n else 0.0


class MeasurementModel(ABC):
    """Interface for anything that predicts a measured quantity from orbit elements."""

    @abstractmethod
    def predict(self, elements: OrbitElements, stations: dict, measurements: list[MeasurementSet]):
        """Return one predicted array per measurement set."""

    @abstractmethod
    def residuals(self, elements, stations, measurements, nuisances):
        """Return a flat weighted residual vector and the per-set predictions."""


class DopplerModel(MeasurementModel):
    """Doppler from SGP4 with one constant (optionally drifting) bias per observation."""

    def __init__(self, orbit: OrbitModel, fc_hz: float, light_time: bool = False):
        self.orbit = orbit
        self.fc_hz = float(fc_hz)
        self.light_time = bool(light_time)
        self._jd_cache: dict[tuple[dt.datetime, ...], tuple[np.ndarray, np.ndarray]] = {}

    def _jd(self, times: list[dt.datetime]) -> tuple[np.ndarray, np.ndarray]:
        key = tuple(times)
        if key not in self._jd_cache:
            self._jd_cache[key] = jd_split_many(times)
        return self._jd_cache[key]

    def predict(self, elements: OrbitElements, stations: dict,
                measurements: list[MeasurementSet]) -> list[np.ndarray]:
        out = []
        for ms in measurements:
            station = stations[ms.station_name]
            jd, fr = self._jd(ms.times_utc)
            fd = self.orbit.doppler(elements, station, jd, fr, self.fc_hz)
            if self.light_time:
                # Retardation: evaluate the state at the emission time
                # t_emit = t_recv - R/c (one fixed-point iteration).
                rng, _ = self.orbit.range_rate(elements, station, jd, fr)
                jd2, fr2 = jd_split_many(
                    [t - dt.timedelta(seconds=float(r) / C_LIGHT)
                     for t, r in zip(ms.times_utc, rng, strict=True)]
                )
                fd = self.orbit.doppler(elements, station, jd2, fr2, self.fc_hz)
            out.append(np.asarray(fd, dtype=float))
        return out

    def residuals(self, elements: OrbitElements, stations: dict,
                  measurements: list[MeasurementSet], nuisances: dict) -> tuple[np.ndarray, dict]:
        """Weighted residuals ``(fd_meas - b_obs - fd_pred)/sigma``.

        ``nuisances`` maps observation id -> {"bias_hz": float, "drift_hz_s": float}.
        """
        pred_all = self.predict(elements, stations, measurements)
        chunks = []
        for ms, pred in zip(measurements, pred_all, strict=True):
            nz = nuisances.get(ms.observation_id, {})
            bias = float(nz.get("bias_hz", 0.0))
            drift = float(nz.get("drift_hz_s", 0.0))
            model = bias + drift * (ms.t_rel_s - ms.t_mid_s())
            sigma = np.maximum(ms.sigma_hz, 1e-12)
            chunks.append((ms.fd_hz - model - pred) / sigma)
        return np.concatenate(chunks) if chunks else np.zeros(0), dict(zip(
            [ms.observation_id for ms in measurements], pred_all, strict=True))


class SymbolRateModel(MeasurementModel):
    """Documented stub for future measurements (thesis-author idea).

    A symbol/timing-rate measurement constrains the range-rate through the
    same ``-fc * dR/dt / c`` relation; implementing it only requires filling
    :meth:`predict` with the appropriate scaling and reusing the residual
    plumbing.  It is intentionally not part of the current fit.
    """

    def predict(self, elements, stations, measurements):  # pragma: no cover - stub
        raise NotImplementedError("symbol-rate measurements are a future extension")

    def residuals(self, elements, stations, measurements, nuisances):  # pragma: no cover
        raise NotImplementedError("symbol-rate measurements are a future extension")

"""SGP4 wrapper that builds satellite states from *continuous* element values.

The fit never goes through TLE-text rounding: element values are handed to
``sgp4init`` directly. Positions/velocities (TEME, km & km/s) are converted to
the topocentric range/range-rate used by the measurement model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sgp4.api import WGS72, Satrec

from .frames import (
    C_LIGHT,
    geodetic_to_ecef,
    teme_to_ecef_states,
)
from .tle import TLE

__all__ = ["OrbitElements", "OrbitModel"]

_XPDOTP = 1440.0 / (2.0 * math.pi)  # rev/day -> rad/min
_DEG = math.pi / 180.0


@dataclass(frozen=True)
class OrbitElements:
    """Mean Keplerian elements at the TLE epoch (angles in degrees, n in rev/day)."""

    n: float
    ecc: float
    i: float
    raan: float
    argp: float
    M: float

    def as_tuple(self) -> tuple[float, ...]:
        return (self.n, self.ecc, self.i, self.raan, self.argp, self.M)


class OrbitModel:
    """Build SGP4 states for arbitrary element values anchored at a fixed epoch."""

    def __init__(self, base_tle: TLE):
        self.base_tle = base_tle
        self._base = Satrec.twoline2rv(base_tle.line1, base_tle.line2)
        self._whichconst = WGS72
        self._opsmode = getattr(self._base, "operationmode", "i")
        self._satnum = int(self._base.satnum_str)
        self._epoch_days = (self._base.jdsatepoch + self._base.jdsatepochF) - 2433281.5
        self._bstar = self._base.bstar
        self._ndot = self._base.ndot
        self._nddot = self._base.nddot
        self.base_elements = OrbitElements(
            n=self._base.no_kozai * _XPDOTP,
            ecc=self._base.ecco,
            i=self._base.inclo / _DEG,
            raan=self._base.nodeo / _DEG,
            argp=self._base.argpo / _DEG,
            M=self._base.mo / _DEG,
        )

    # -- construction ------------------------------------------------------
    def make_satrec(self, el: OrbitElements) -> Satrec:
        sat = Satrec()
        sat.sgp4init(
            self._whichconst,
            self._opsmode,
            self._satnum,
            self._epoch_days,
            self._bstar,
            self._ndot,
            self._nddot,
            el.ecc,
            el.argp * _DEG,
            el.i * _DEG,
            el.M * _DEG,
            el.n / _XPDOTP,
            el.raan * _DEG,
        )
        return sat

    def to_tle(self, el: OrbitElements) -> TLE:
        return self.base_tle.with_elements(i=el.i, raan=el.raan, ecc=el.ecc, argp=el.argp,
                                           M=el.M, n=el.n)

    # -- propagation -------------------------------------------------------
    @staticmethod
    def _propagate(sat: Satrec, jd: float, fr: float):
        err, r, v = sat.sgp4(jd, fr)
        if err != 0:
            return None, None
        return np.asarray(r, dtype=float), np.asarray(v, dtype=float)

    def state_teme(self, el: OrbitElements, jd: float, fr: float):
        """TEME state (km, km/s) at the Julian date (jd, fr)."""
        sat = self.make_satrec(el)
        return self._propagate(sat, float(jd), float(fr))

    def range_rate(self, el: OrbitElements, station, jd, fr):
        """Return range (m) and range-rate (m/s) arrays for a station.

        ``jd``/``fr`` are the Julian-date split (see :func:`timeutil.jd_split`).
        """
        sat = self.make_satrec(el)
        sta_ecef = geodetic_to_ecef(station.lat_deg, station.lon_deg, station.alt_m)
        jd = np.atleast_1d(np.asarray(jd, dtype=float))
        fr = np.atleast_1d(np.asarray(fr, dtype=float))
        err, r_teme, v_teme = sat.sgp4_array(jd, fr)
        bad = np.asarray(err) != 0
        r_ecef, v_ecef = teme_to_ecef_states(r_teme, v_teme, jd, fr)
        los_m = r_ecef * 1000.0 - sta_ecef
        rng = np.linalg.norm(los_m, axis=1)
        u = los_m / rng[:, None]
        rr = np.sum(v_ecef * 1000.0 * u, axis=1)
        rng[bad] = np.nan
        rr[bad] = np.nan
        return rng, rr

    def doppler(self, el: OrbitElements, station, jd, fr, fc_hz: float):
        """Predicted Doppler (Hz), positive approaching: fd = -fc*dR/dt/c."""
        _, rr = self.range_rate(el, station, jd, fr)
        return -fc_hz * rr / C_LIGHT

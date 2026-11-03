"""Time helpers: robust UTC parsing and Julian-date conversion."""

from __future__ import annotations

import datetime as _dt

__all__ = ["UTC", "parse_iso", "to_utc", "jd_split", "jd_float", "from_jd", "unix_seconds"]

UTC = _dt.UTC


def parse_iso(text: str) -> _dt.datetime:
    """Parse an ISO-8601 timestamp; tolerate a trailing 'Z', naive input is treated as UTC."""
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = _dt.datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def to_utc(dt: _dt.datetime) -> _dt.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def jd_split(dt: _dt.datetime) -> tuple[float, float]:
    """Return (jd, fraction) for a datetime (UTC).

    The Julian date of midnight (``X.5``) and the fraction of the day are
    computed separately so that no precision is lost: a full Julian date as a
    float64 only resolves ~47 us, which is ~0.3 m of satellite motion and
    therefore visible in the Doppler.
    """
    dt = to_utc(dt)
    jd = float(dt.date().toordinal() + 1721424.5)
    secs = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond * 1e-6
    return jd, secs / 86400.0


def jd_split_many(dts):
    """Vectorised :func:`jd_split` for a sequence of datetimes."""
    import numpy as np

    dts = [to_utc(d) for d in dts]
    jd = np.array([float(d.date().toordinal() + 1721424.5) for d in dts])
    secs = np.array([
        d.hour * 3600.0 + d.minute * 60.0 + d.second + d.microsecond * 1e-6 for d in dts
    ])
    return jd, secs / 86400.0


def jd_float(dt: _dt.datetime) -> float:
    jd, fr = jd_split(dt)
    return jd + fr


def split_jd(jd_total):
    """Split a full Julian date into the (jd, fr) pair expected by ``sgp4``.

    Only used for display/round-tripping: a full Julian date stored as a
    float64 resolves only ~47 us, so propagation should use :func:`jd_split`.
    """
    import numpy as np

    arr = np.asarray(jd_total, dtype=float)
    jd = np.floor(arr - 0.5) + 0.5
    fr = arr - jd
    if np.isscalar(jd_total) or np.ndim(jd_total) == 0:
        return float(jd), float(fr)
    return jd, fr


def from_jd(jd: float) -> _dt.datetime:
    """Julian date -> UTC datetime."""
    return _dt.datetime(1970, 1, 1, tzinfo=UTC) + _dt.timedelta(days=jd - 2440587.5)


def unix_seconds(dt: _dt.datetime) -> float:
    return to_utc(dt).timestamp()

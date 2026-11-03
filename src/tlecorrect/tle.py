"""TLE parsing, formatting and checksum utilities.

Implemented from scratch (no ``sgp4`` dependency here) so that round-trip and
checksum behaviour can be unit-tested in isolation.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, replace

__all__ = ["TLE", "parse_tle", "tle_checksum", "epoch_to_datetime", "datetime_to_epoch"]


def tle_checksum(line: str) -> int:
    """Modulo-10 checksum of a TLE line (first 68 characters, '-' counts as 1)."""
    if len(line) < 68:
        raise ValueError("TLE line must be at least 68 characters long")
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def epoch_to_datetime(epoch: str) -> _dt.datetime:
    """Convert a TLE epoch field ``YYDDD.DDDDDDDD`` to a timezone-aware UTC datetime."""
    epoch = epoch.strip()
    if len(epoch) < 8 or epoch[5] != ".":
        raise ValueError(f"bad TLE epoch field: {epoch!r}")
    yy = int(epoch[:2])
    doy = float(epoch[2:])
    year = 2000 + yy if yy <= 56 else 1900 + yy
    whole = int(math.floor(doy))
    frac = doy - whole
    base = _dt.datetime(year, 1, 1, tzinfo=_dt.UTC) + _dt.timedelta(days=whole - 1)
    return base + _dt.timedelta(seconds=frac * 86400.0)


def datetime_to_epoch(dt: _dt.datetime) -> str:
    """Format a datetime as a TLE epoch field ``YYDDD.DDDDDDDD`` (UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.UTC)
    dt = dt.astimezone(_dt.UTC)
    start = _dt.datetime(dt.year, 1, 1, tzinfo=_dt.UTC)
    doy = (dt - start).total_seconds() / 86400.0 + 1.0
    yy = dt.year % 100
    return f"{yy:02d}{doy:012.8f}".replace(" ", "0")


def _parse_exp_field(field: str) -> float:
    """Parse a TLE assumed-decimal-point exponent field, e.g. ' 38013-4' -> 0.38013e-4."""
    field = field.strip()
    if not field:
        return 0.0
    sign = 1.0
    if field[0] in "+-":
        sign = -1.0 if field[0] == "-" else 1.0
        field = field[1:]
    mant = ""
    exp = ""
    for ch in field:
        if ch in "+-" and mant:
            exp = ch + field[field.index(ch) + 1 :]
            break
        mant += ch
    if not exp:
        return sign * float("0." + mant) if mant else 0.0
    exp_sign = -1 if exp[0] == "-" else 1
    exp_val = int(exp[1:]) if len(exp) > 1 else 0
    return sign * float("0." + mant) * (10.0 ** (exp_sign * exp_val))


def _parse_decimal(field: str) -> float:
    """Parse a TLE implied-decimal field (optional sign, leading '.'), e.g. ' .00000027'."""
    s = field.strip()
    if not s:
        return 0.0
    sign = 1.0
    if s[0] in "+-":
        sign = -1.0 if s[0] == "-" else 1.0
        s = s[1:]
    if s.startswith("."):
        s = "0" + s
    return sign * float(s)


def _format_exp_field(value: float, width: int) -> str:
    """Format a value in the TLE assumed-decimal-point form (e.g. 0.38013e-4 -> ' 38013-4')."""
    if value == 0.0:
        return " " + "0" * (width - 2) + "+0"
    sign = "-" if value < 0 else " "
    value = abs(value)
    exp = int(math.floor(math.log10(value)))
    mant = value / (10.0**exp)
    # mantissa in [0.1, 1)
    while mant < 0.1:
        mant *= 10
        exp -= 1
    while mant >= 1.0:
        mant /= 10
        exp += 1
    digits = int(round(mant * (10 ** (width - 3))))
    if digits >= 10 ** (width - 2):
        digits //= 10
        exp += 1
    exp_sign = "-" if exp < 0 else "+"
    exp_str = f"{exp_sign}{abs(exp):d}"
    digits_str = f"{digits:0{width - 2}d}"
    return sign + digits_str + exp_str


@dataclass(frozen=True)
class TLE:
    """A two-line element set with parsed element values."""

    line1: str
    line2: str
    name: str = ""

    # parsed
    satnum: int = 0
    epoch: _dt.datetime = None  # type: ignore[assignment]
    n: float = 0.0  # mean motion, rev/day
    i: float = 0.0  # inclination, deg
    raan: float = 0.0  # deg
    ecc: float = 0.0  # 0..1
    argp: float = 0.0  # deg
    M: float = 0.0  # deg
    bstar: float = 0.0
    ndot: float = 0.0
    nddot: float = 0.0
    revnum: int = 0

    def __post_init__(self) -> None:
        if self.epoch is None:
            raise ValueError("TLE requires an epoch")

    # -- derived -----------------------------------------------------------
    @property
    def semi_major_axis_m(self) -> float:
        mu = 3.986004418e14
        n_rad = self.n * 2.0 * math.pi / 86400.0
        return (mu / n_rad**2) ** (1.0 / 3.0)

    @property
    def period_s(self) -> float:
        return 86400.0 / self.n

    # -- serialisation -----------------------------------------------------
    def with_elements(self, *, i=None, raan=None, ecc=None, argp=None, M=None, n=None) -> TLE:
        """Return a new TLE with selected elements replaced (line2 reformatted, checksum fixed)."""
        i = self.i if i is None else i
        raan = self.raan if raan is None else raan
        ecc = self.ecc if ecc is None else ecc
        argp = self.argp if argp is None else argp
        M = self.M if M is None else M
        n = self.n if n is None else n
        # store values rounded to the TLE field precision so the object is
        # self-consistent with the emitted text
        i = round(i, 4)
        raan = round(raan % 360.0, 4)
        ecc = round(ecc, 7)
        argp = round(argp % 360.0, 4)
        M = round(M % 360.0, 4)
        n = round(n, 8)
        line2 = _format_line2(self.satnum, i, raan, ecc, argp, M, n, self.revnum)
        return replace(self, line2=line2, i=i, raan=raan, ecc=ecc, argp=argp, M=M, n=n)

    def as_text(self, with_name: bool = True) -> str:
        out = []
        if with_name and self.name:
            out.append(f"# {self.name}")
        out.append(self.line1)
        out.append(self.line2)
        return "\n".join(out) + "\n"

    @property
    def norad_id(self) -> int:
        return self.satnum

    def element_str(self) -> str:
        return (
            f"epoch={self.epoch.isoformat()} n={self.n:.8f} rev/day i={self.i:.4f} "
            f"raan={self.raan:.4f} ecc={self.ecc:.7f} argp={self.argp:.4f} "
            f"M={self.M:.4f} bstar={self.bstar:.5e}"
        )


def _format_line2(satnum, i, raan, ecc, argp, M, n, revnum) -> str:
    ecc7 = f"{ecc:.7f}"[2:]
    body = (
        f"2 {satnum:5d} {i:8.4f} {raan:8.4f} {ecc7:>7s} "
        f"{argp:8.4f} {M:8.4f} {n:11.8f}{revnum:5d}"
    )
    body = body[:68].ljust(68)
    return f"{body}{tle_checksum(body)}"


def parse_tle(text: str) -> TLE:
    """Parse TLE text (1 or 2 lines, optional name/comment lines)."""
    lines = [ln.rstrip("\r\n") for ln in text.strip().splitlines()]
    lines = [ln for ln in lines if ln.strip()]
    line1 = line2 = None
    name = ""
    for ln in lines:
        if ln.startswith("1 "):
            line1 = ln
        elif ln.startswith("2 "):
            line2 = ln
        elif ln.startswith("#"):
            if not name:
                name = ln.lstrip("# ").strip()
        elif not name and not ln[0].isdigit():
            name = ln.strip()
    if line1 is None or line2 is None:
        raise ValueError("TLE text must contain a line starting with '1 ' and one with '2 '")
    if len(line1) < 68:
        line1 = line1.ljust(68)
    if len(line2) < 68:
        line2 = line2.ljust(68)
    _verify_checksum(line1, "line1")
    _verify_checksum(line2, "line2")

    satnum = int(line1[2:7])
    epoch = epoch_to_datetime(line1[18:32])
    ndot = _parse_decimal(line1[33:43])
    nddot = _parse_exp_field(line1[44:52])
    bstar = _parse_exp_field(line1[53:61])

    i = float(line2[8:16])
    raan = float(line2[17:25])
    ecc = float("0." + line2[26:33].strip())
    argp = float(line2[34:42])
    M = float(line2[43:51])
    n = float(line2[52:63])
    revnum = int(line2[63:68]) if line2[63:68].strip().isdigit() else 0

    return TLE(
        line1=line1,
        line2=line2,
        name=name,
        satnum=satnum,
        epoch=epoch,
        n=n,
        i=i,
        raan=raan,
        ecc=ecc,
        argp=argp,
        M=M,
        bstar=bstar,
        ndot=ndot,
        nddot=nddot,
        revnum=revnum,
    )


def _verify_checksum(line: str, which: str) -> None:
    if not line[68:69].isdigit():
        raise ValueError(f"{which}: missing checksum digit")
    expected = int(line[68])
    actual = tle_checksum(line)
    if expected != actual:
        raise ValueError(f"{which}: checksum mismatch (got {expected}, expected {actual})")

"""Minimal, self-contained SigMF reader (metadata + chunked I/Q access).

Only the subset used by recorded satellite passes is implemented, but it is
checked against the reference ``sigmf`` package in the unit tests.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np

from .timeutil import parse_iso

__all__ = ["SigMFRecording", "DTYPE_MAP", "itemsize_for"]

# SigMF core datatype -> numpy dtype (little-endian)
DTYPE_MAP: dict[str, str] = {
    "cf32_le": "complex64",
    "cf64_le": "complex128",
    "cf32_be": ">c8",
    "ci16_le": "int16",
    "ci16_be": ">i2",
    "ci8": "int8",
    "cu8": "uint8",
    "cu16_le": "uint16",
    "rf32_le": "float32",
    "rf64_le": "float64",
}


def itemsize_for(datatype: str) -> int:
    if datatype not in DTYPE_MAP:
        raise ValueError(f"unsupported SigMF datatype {datatype!r}")
    return np.dtype(DTYPE_MAP[datatype]).itemsize


class SigMFRecording:
    """A SigMF recording (meta + data pair)."""

    def __init__(self, meta_path: str | Path, data_path: str | Path | None = None):
        self.meta_path = Path(meta_path)
        if not self.meta_path.exists():
            raise FileNotFoundError(f"SigMF meta not found: {self.meta_path}")
        with open(self.meta_path, encoding="utf-8-sig") as fh:
            self.meta = json.load(fh)
        g = self.meta.get("global", {})
        if "core:datatype" not in g:
            raise ValueError(f"{self.meta_path}: missing global core:datatype")
        self.datatype = g["core:datatype"]
        self.sample_rate = float(g.get("core:sample_rate", 0.0))
        self.offset = int(g.get("core:offset", 0))
        self.datetime = parse_iso(g["core:datetime"]) if g.get("core:datetime") else None
        self.global_freq = float(g["core:freq"]) if g.get("core:freq") is not None else None
        caps = self.meta.get("captures", [])
        self.captures = caps
        self.capture_freq = None
        if caps and caps[0].get("core:frequency") is not None:
            self.capture_freq = float(caps[0]["core:frequency"])
        self.extension = dict(g)

        if data_path is not None:
            self.data_path = Path(data_path)
        else:
            stem = self.meta_path.name
            if stem.endswith(".sigmf-meta"):
                stem = stem[: -len(".sigmf-meta")]
            self.data_path = self.meta_path.with_name(stem + ".sigmf-data")
        if not self.data_path.exists():
            raise FileNotFoundError(f"SigMF data not found: {self.data_path}")
        self.itemsize = itemsize_for(self.datatype)
        self.num_samples = self.data_path.stat().st_size // self.itemsize
        self._memmap = np.memmap(
            self.data_path, dtype=np.dtype(DTYPE_MAP[self.datatype]), mode="r"
        )

    # -- accessors ---------------------------------------------------------
    @property
    def center_freq(self) -> float | None:
        """SDR center frequency, in priority order: capture, global core:freq."""
        if self.capture_freq is not None:
            return self.capture_freq
        return self.global_freq

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate if self.sample_rate else 0.0

    @property
    def start_utc(self) -> dt.datetime | None:
        return self.datetime

    def read(self, start: int = 0, count: int | None = None) -> np.ndarray:
        if count is None:
            count = self.num_samples - start
        count = max(0, min(count, self.num_samples - start))
        return np.asarray(self._memmap[start : start + count])

    def read_time_range(self, t0: dt.datetime, t1: dt.datetime) -> np.ndarray:
        """Read the samples whose UTC timestamps fall in [t0, t1)."""
        if self.datetime is None:
            raise ValueError("recording has no core:datetime; cannot slice by time")
        s0 = int(round((t0 - self.datetime).total_seconds() * self.sample_rate))
        s1 = int(round((t1 - self.datetime).total_seconds() * self.sample_rate))
        s0 = max(0, s0)
        s1 = min(self.num_samples, s1)
        return self.read(s0, s1 - s0)

    def time_of_sample(self, idx: int) -> dt.datetime:
        if self.datetime is None:
            raise ValueError("recording has no core:datetime")
        return self.datetime + dt.timedelta(seconds=idx / self.sample_rate)

    def summary(self) -> dict:
        return {
            "meta_path": str(self.meta_path),
            "data_path": str(self.data_path),
            "datatype": self.datatype,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "duration_s": self.duration,
            "start_utc": self.datetime.isoformat() if self.datetime else None,
            "center_freq": self.center_freq,
        }

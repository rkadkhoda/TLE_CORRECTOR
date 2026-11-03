"""Pluggable Doppler (instantaneous frequency) extractors.

Two estimators are provided:

* ``STFTRidgeExtractor`` — STFT ridge tracking with a Viterbi continuity
  constraint and parabolic peak interpolation.
* ``PhaseSlopeExtractor`` — frequency from the unwrapped phase over a window
  (Kay weighted phase-difference estimator) around a coarse frequency.

Both measure the *baseband* frequency and convert it to Doppler with
``f_d = f_bb + f_lo - f_c`` (positive approaching).

Mode handling
-------------
``tone``
    The signal is a CW carrier; the baseband frequency is the carrier.
``fm_center``
    The signal is a tone-modulated FM carrier.  The carrier is the mean
    instantaneous frequency, i.e. the mean over an integer number of
    modulation periods.  The spectral ridge locks onto a *sideband* (the
    strongest FM line is generally not the carrier), therefore the coarse
    reference is obtained from the mean of the wrapped instantaneous
    frequency instead of from the spectral peak, and the averaging window is
    an integer number of modulation periods when the modulation frequency is
    known.
"""

from __future__ import annotations

import datetime as dt
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft as _fft

__all__ = [
    "Measurement",
    "ExtractionResult",
    "Extractor",
    "STFTRidgeExtractor",
    "PhaseSlopeExtractor",
    "AutoExtractor",
    "make_extractor",
    "fbb_to_fd",
    "spectrogram",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class Measurement:
    time_utc: dt.datetime
    t_rel_s: float  # seconds from recording start
    fd_hz: float  # Doppler, positive approaching
    sigma_hz: float  # 1-sigma (CRLB approximation)
    snr_db: float
    quality: str = "ok"  # ok | flagged | dropped
    reason: str = ""


@dataclass
class ExtractionResult:
    measurements: list[Measurement] = field(default_factory=list)
    method: str = ""
    mode: str = ""
    warnings: list[str] = field(default_factory=list)
    spectrogram: dict | None = None

    def times(self) -> np.ndarray:
        return np.array([m.t_rel_s for m in self.measurements])

    def fd(self) -> np.ndarray:
        return np.array([m.fd_hz for m in self.measurements])

    def sigma(self) -> np.ndarray:
        return np.array([m.sigma_hz for m in self.measurements])

    def ok_mask(self) -> np.ndarray:
        return np.array([m.quality == "ok" for m in self.measurements])

    def used(self) -> list[Measurement]:
        """Measurements the fit may use (quality ``ok``).

        ``flagged`` windows (aliasing risk or low-SNR fallback) are kept in the
        results for inspection but are not trusted by the fit; ``dropped``
        windows have no usable signal at all.
        """
        return [m for m in self.measurements if m.quality == "ok"]


def fbb_to_fd(f_bb_hz, flo_hz: float, fc_hz: float):
    """Convert baseband frequency to Doppler: f_d = f_bb + f_lo - f_c."""
    return f_bb_hz + flo_hz - fc_hz


# ---------------------------------------------------------------------------
# Spectrogram helpers
# ---------------------------------------------------------------------------
def spectrogram(x: np.ndarray, fs: float, nfft: int = 4096, hop: int | None = None,
                block: int = 4096):
    """Return (freqs, times, power) of the STFT of a *complex* baseband signal.

    The full signed spectrum (-fs/2 .. +fs/2) is kept: a complex baseband
    signal is not symmetric, so the ``rfft`` half-spectrum must not be used.
    """
    if hop is None:
        hop = max(1, nfft // 4)
    w = np.hanning(nfft)
    n_frames = 1 + max(0, (len(x) - nfft) // hop)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    power = np.empty((n_frames, len(freqs)), dtype=np.float32)
    wsum = float(np.sum(w**2))
    xa = np.asarray(x)
    for start in range(0, n_frames, block):
        stop = min(n_frames, start + block)
        idx = np.arange(start, stop)[:, None] * hop + np.arange(nfft)[None, :]
        seg = xa[idx].astype(np.complex64) * w
        spec = np.fft.fftshift(_fft(seg, axis=1), axes=1)
        power[start:stop] = (np.abs(spec) ** 2) / wsum
    times = (np.arange(n_frames) * hop + nfft / 2.0) / fs
    return freqs, times, power


def _signed_band(freqs: np.ndarray, f_center: float, half_width_hz: float) -> np.ndarray:
    """Boolean mask of the bins that belong to the band around ``f_center``.

    For complex signals the band has an image at ``-f_center``.
    """
    return (np.abs(freqs - f_center) <= half_width_hz) | (
        np.abs(freqs + f_center) <= half_width_hz
    )


def _viterbi_ridge(freqs, mag_db, max_jump_hz, penalty=0.3, decim=8):
    """Viterbi tracking of a continuous ridge (decimated frequency axis, vectorised).

    ``freqs`` must be the signed frequency axis of the full complex spectrum,
    i.e. 0 .. +fs/2 followed by -fs/2 .. 0.
    """
    n_frames = mag_db.shape[0]
    df_fine = abs(freqs[1] - freqs[0])
    decim = max(1, int(decim))
    n_groups = len(freqs) // decim
    if n_groups < 2:
        decim = 1
        n_groups = len(freqs)
    magc = mag_db[:, : n_groups * decim].reshape(n_frames, n_groups, decim).max(axis=2)
    df_c = df_fine * decim
    b = max(1, int(round(max_jump_hz / df_c)))
    b = min(b, n_groups - 1)
    off = np.arange(-b, b + 1)
    pen = penalty * off.astype(np.float64) ** 2
    dp = -magc[0].astype(np.float64)
    back = np.zeros((n_frames, n_groups), dtype=np.int32)
    idx = np.arange(n_groups)
    for m in range(1, n_frames):
        pad = np.full(b, 1e18)
        dp_pad = np.concatenate([pad, dp, pad])
        sw = np.lib.stride_tricks.sliding_window_view(dp_pad, 2 * b + 1)
        cand = sw + pen
        j = np.argmin(cand, axis=1)
        dp = cand[idx, j] - magc[m]
        back[m] = idx - b + j
    path_c = np.empty(n_frames, dtype=np.int32)
    path_c[-1] = int(np.argmin(dp))
    for m in range(n_frames - 1, 0, -1):
        path_c[m - 1] = back[m, path_c[m]]
    path = np.empty(n_frames, dtype=np.int32)
    for m in range(n_frames):
        g = int(path_c[m])
        lo = g * decim
        hi = min(len(freqs), lo + decim)
        path[m] = lo + int(np.argmax(mag_db[m, lo:hi]))
    return path


def _parabolic(freqs, row, k):
    if k <= 0 or k >= len(row) - 1:
        return freqs[k]
    a, b, c = row[k - 1], row[k], row[k + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return freqs[k]
    frac = 0.5 * (a - c) / denom
    frac = max(-0.5, min(0.5, frac))
    return freqs[k] + frac * (freqs[1] - freqs[0])


def _kay_weights(n: int) -> np.ndarray:
    """Kay's weights for a phase-difference frequency estimator (kept for reference)."""
    k = np.arange(n - 1)
    return 3.0 * n / (n * n - 1) * (1.0 - ((2.0 * k - (n - 2)) / n) ** 2)


def _phase_slope_hz(ph: np.ndarray, fs: float) -> float:
    """Frequency (Hz) from an unwrapped phase ramp (centred least-squares slope).

    For a phase model ``phi_k = 2*pi*f*k/fs + theta`` the centred least-squares
    slope is unbiased and reaches the CRLB of a complex tone:
    ``var(f) = 6 fs^2 / ((2 pi)^2 SNR N (N^2 - 1))``.
    """
    n = len(ph)
    if n < 3:
        return 0.0
    k = np.arange(n, dtype=float)
    km = k - k.mean()
    den = float(np.sum(km * km))
    if den <= 0:
        return 0.0
    num = float(np.sum(km * ph))
    return num * fs / (2.0 * np.pi * den)


def _estimate_snr(seg: np.ndarray, fs: float, f_center: float,
                  half_width_hz: float) -> tuple[float, float]:
    """(total SNR dB, in-band SNR dB) of a window.

    The noise floor is taken from the periodogram outside the signal band.
    *total* SNR is signal power over noise power in the full sampling band;
    *in-band* SNR is signal power over the noise power inside the signal band.
    """
    n = len(seg)
    if n < 16:
        return -99.0, -99.0
    w = np.hanning(n)
    spec = np.fft.fftshift(_fft(np.asarray(seg, dtype=np.complex64) * w))
    psd = (np.abs(spec) ** 2) / (fs * float(np.sum(w**2)))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / fs))
    band = _signed_band(freqs, f_center, half_width_hz)
    if band.all():
        band[:] = False
    # Periodogram bins of complex white Gaussian noise are exponentially
    # distributed: median = ln(2) * mean.  Dividing by ln(2) gives an
    # unbiased, outlier-robust noise floor (using the raw median would bias
    # the SNR by +1.6 dB).
    if (~band).any():
        noise_psd = float(np.median(psd[~band])) / math.log(2.0)
    else:
        noise_psd = float(np.median(psd)) / math.log(2.0)
    noise_psd = max(noise_psd, 1e-30)
    total = float(np.mean(np.abs(seg) ** 2))
    noise_total = noise_psd * fs
    signal = max(total - noise_total, 0.0)
    snr_total = 10.0 * math.log10(max(signal, 1e-30) / noise_total)
    df = fs / n
    noise_in_band = max(noise_psd * float(band.sum()) * df, 1e-30)
    snr_band = 10.0 * math.log10(max(signal, 1e-30) / noise_in_band)
    return snr_total, snr_band


def _wrapped_inst_freq(x: np.ndarray, fs: float) -> np.ndarray:
    """Wrapped instantaneous frequency (Hz) from consecutive phase differences."""
    if len(x) < 2:
        return np.zeros(0)
    d = np.angle(x[1:] * np.conj(x[:-1]))
    return d * fs / (2.0 * np.pi)


def _circular_freq(z: np.ndarray, fs: float) -> tuple[float, float]:
    """Circular-mean lag-1 frequency and its robust 1-sigma (Hz).

    The phase of the mean unit phasor is the circular mean of the phase
    differences.  Unlike averaging wrapped phase differences directly, this is
    immune to 2*pi phase wraps, which is what makes the estimator usable at low
    SNR where fades push individual differences past +-pi.  ``sigma`` is a
    conservative estimate obtained from the resultant length (von Mises
    approximation), so it grows when the spread of the differences is large
    (low SNR or strong modulation).
    """
    if len(z) < 2:
        return 0.0, float("inf")
    d = np.angle(z[1:] * np.conj(z[:-1]))
    c = np.mean(np.exp(1j * d))
    f = float(np.angle(c)) * fs / (2.0 * np.pi)
    res = np.angle(np.exp(1j * (d - np.angle(c))))
    var_d = float(np.mean(res**2))
    sigma = math.sqrt(max(var_d, 1e-30) / len(d)) * fs / (2.0 * np.pi)
    return f, float(sigma)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
class Extractor(ABC):
    method = "base"

    def __init__(self, **opts):
        self.opts = opts

    @abstractmethod
    def extract(self, x: np.ndarray, fs: float, start_utc: dt.datetime, fc_hz: float,
                flo_hz: float, mode: str = "tone", **kw) -> ExtractionResult:
        ...


class STFTRidgeExtractor(Extractor):
    """Ridge of the STFT magnitude, tracked with a Viterbi continuity constraint."""

    method = "stft"

    def extract(self, x, fs, start_utc, fc_hz, flo_hz, mode="tone", nfft=4096, hop=None,
                window_s=None, max_jump_hz=400.0, min_snr_db=-5.0,
                signal_half_width_hz=None, **kw) -> ExtractionResult:
        x = np.asarray(x)
        if window_s is not None:
            nfft = int(round(window_s * fs))
        if hop is None:
            hop = max(1, nfft // 4)
        freqs, times, power = spectrogram(x, fs, nfft=nfft, hop=hop)
        mag = np.sqrt(power)
        decim = max(1, int(round(60.0 / max(freqs[1] - freqs[0], 1e-9))))
        path = _viterbi_ridge(freqs, 10.0 * np.log10(np.maximum(mag, 1e-30)),
                              max_jump_hz=max_jump_hz, decim=decim)
        nyq = fs / 2.0
        edge = 0.05 * fs
        res = ExtractionResult(method=self.method, mode=mode)
        for m in range(len(times)):
            row = power[m]
            k = int(path[m])
            f_bb = _parabolic(freqs, np.sqrt(row), k)
            noise = float(np.median(power[m])) / math.log(2.0)
            peak = float(row.max())
            # peak-to-noise in a single STFT bin carries the coherent
            # integration gain; remove it so that ``snr_db`` is the
            # per-sample SNR and ``min_snr_db`` means the same in both
            # extractors.
            gain_db = 10.0 * math.log10(max(1.0, 0.96 * nfft))
            snr_db = 10.0 * math.log10(max(peak, 1e-30) / max(noise, 1e-30)) - gain_db
            quality = "ok"
            reason = ""
            if snr_db < min_snr_db:
                quality, reason = "dropped", f"low SNR ({snr_db:.1f} dB)"
            elif abs(f_bb) > nyq - edge:
                quality = "flagged"
                reason = "near Nyquist edge"
            sigma = _crlb_sigma(fs, nfft, snr_db)
            res.measurements.append(
                Measurement(
                    time_utc=start_utc + dt.timedelta(seconds=float(times[m])),
                    t_rel_s=float(times[m]),
                    fd_hz=float(fbb_to_fd(f_bb, flo_hz, fc_hz)),
                    sigma_hz=float(sigma),
                    snr_db=float(snr_db),
                    quality=quality,
                    reason=reason,
                )
            )
        res.spectrogram = {"freqs": freqs, "times": times, "power": power}
        return res


class PhaseSlopeExtractor(Extractor):
    """Frequency from the unwrapped phase over each window (Kay estimator)."""

    method = "phase"

    def extract(self, x, fs, start_utc, fc_hz, flo_hz, mode="tone", window_s=0.1,
                hop_s=None, coarse_result: ExtractionResult | None = None,
                min_snr_db=-5.0, mod_freq_hz=None, signal_half_width_hz=None,
                use_stft_coarse=None, edge_fraction=0.95, edge_frac_max=0.002,
                edge_min_snr_db=10.0, phase_ls_min_snr_db=7.0, **kw) -> ExtractionResult:
        x = np.asarray(x)
        if use_stft_coarse is None:
            use_stft_coarse = (mode != "fm_center")
        if mod_freq_hz and mod_freq_hz > 0 and window_s > 0:
            # use an integer number of modulation periods so the modulation
            # averages out exactly in the least-squares slope
            k = max(1, int(round(window_s * mod_freq_hz)))
            window_s = k / mod_freq_hz
        n = int(round(window_s * fs))
        if n < 8:
            raise ValueError("extraction window is too short")
        if hop_s is None:
            hop_s = window_s
        hop = max(1, int(round(hop_s * fs)))
        starts = np.arange(0, max(1, len(x) - n + 1), hop)
        nyq = fs / 2.0
        if signal_half_width_hz is None:
            signal_half_width_hz = max(200.0, 0.02 * fs)
        inst = _wrapped_inst_freq(x, fs)
        res = ExtractionResult(method=self.method, mode=mode)
        for s0 in starts:
            seg = np.asarray(x[s0 : s0 + n], dtype=np.complex64)
            t = np.arange(n) / fs
            center = (s0 + n / 2.0) / fs
            i0 = min(max(s0 - 1, 0), max(0, len(inst) - 1))
            i1 = min(i0 + n - 1, len(inst))
            inst_win = inst[i0:i1] if i1 > i0 else np.zeros(1)
            if use_stft_coarse and coarse_result is not None:
                f_coarse = _coarse_freq(coarse_result, center) - (flo_hz - fc_hz)
            else:
                # carrier estimate: the modulation averages out over an
                # integer number of periods, and the residual error of the
                # mixing reference cancels in the phase-slope refinement.
                # The circular mean is used so that 2*pi wraps caused by noise
                # fades do not bias the estimate.
                f_coarse, _ = _circular_freq(seg, fs)
            snr_total, snr_band = _estimate_snr(seg, fs, f_coarse, signal_half_width_hz)
            t0 = t - t[0]
            z = seg * np.exp(-2j * np.pi * f_coarse * t0)
            crlb = _crlb_sigma(fs, n, snr_total, snr_band,
                               bandwidth_hz=2.0 * signal_half_width_hz)
            if snr_total >= phase_ls_min_snr_db:
                ph = np.unwrap(np.angle(z))
                f_res = _phase_slope_hz(ph, fs)
                sigma = crlb
                est_method = "phase-slope"
            else:
                # Below the phase-slope threshold the unwrapped phase is
                # corrupted by 2*pi slips; the circular mean is less accurate
                # but unbiased, and its sigma is taken from the data.
                f_res, sigma_low = _circular_freq(z, fs)
                sigma = max(crlb, sigma_low)
                est_method = "circular"
            f_bb = f_coarse + f_res
            # aliasing risk: the instantaneous frequency must stay clear of
            # Nyquist.  Only meaningful when the signal actually dominates the
            # window; at low SNR the noise alone would trigger the flag.
            frac_edge = float(np.mean(np.abs(inst_win) > edge_fraction * nyq))
            quality = "ok"
            reason = ""
            if snr_total < min_snr_db:
                quality, reason = "dropped", f"low SNR ({snr_total:.1f} dB)"
            elif frac_edge > edge_frac_max and snr_total >= edge_min_snr_db:
                quality = "flagged"
                reason = (f"{100 * frac_edge:.1f}% of the samples have an instantaneous "
                          f"frequency within {100 * (1 - edge_fraction):.0f}% of the Nyquist "
                          f"limit ({nyq:.0f} Hz) - aliasing risk")
            elif est_method == "circular":
                quality = "flagged"
                reason = (f"below the phase-slope threshold ({snr_total:.1f} dB); "
                          f"circular fallback used")
            res.measurements.append(
                Measurement(
                    time_utc=start_utc + dt.timedelta(seconds=float(center)),
                    t_rel_s=float(center),
                    fd_hz=float(fbb_to_fd(f_bb, flo_hz, fc_hz)),
                    sigma_hz=float(sigma),
                    snr_db=float(snr_total),
                    quality=quality,
                    reason=reason,
                )
            )
        return res


class AutoExtractor(Extractor):
    """STFT ridge for detection/preview, phase-slope for the final frequency."""

    method = "auto"

    def extract(self, x, fs, start_utc, fc_hz, flo_hz, mode="tone", **opts) -> ExtractionResult:
        nfft = opts.get("nfft", 4096)
        hop_s = opts.get("hop_s")
        max_jump_hz = opts.get("max_jump_hz", 400.0)
        min_snr_db = opts.get("min_snr_db", -5.0)
        window_s = opts.get("window_s")
        stft = STFTRidgeExtractor()
        coarse = stft.extract(x, fs, start_utc, fc_hz, flo_hz, mode=mode, nfft=nfft,
                              hop=None if hop_s is None else max(1, int(hop_s * fs)),
                              max_jump_hz=max_jump_hz, min_snr_db=min_snr_db)
        if mode == "fm_center" and (window_s is None or window_s < 1.0):
            window_s = 1.0
        if window_s is None:
            window_s = 0.2
        dev = opts.get("deviation_hz") or 0.0
        mod = opts.get("mod_freq_hz") or 0.0
        half = opts.get("signal_half_width_hz")
        if half is None and mode == "fm_center":
            half = float(dev + 2.0 * mod + 100.0)
        refine = PhaseSlopeExtractor().extract(
            x, fs, start_utc, fc_hz, flo_hz, mode=mode, window_s=window_s,
            hop_s=hop_s or window_s, coarse_result=coarse, min_snr_db=min_snr_db,
            **{k: v for k, v in opts.items()
               if k in ("mod_freq_hz", "deviation_hz",
                        "edge_fraction", "edge_frac_max", "edge_min_snr_db",
                        "phase_ls_min_snr_db")},
            signal_half_width_hz=half,
        )
        refine.warnings = list(coarse.warnings)
        refine.spectrogram = coarse.spectrogram
        refine.method = "auto(stft+phase)"
        return refine


def _coarse_freq(coarse_result: ExtractionResult | None, t_center: float) -> float:
    if coarse_result is None or not coarse_result.measurements:
        return 0.0
    t = np.array([m.t_rel_s for m in coarse_result.measurements])
    f = np.array([m.fd_hz for m in coarse_result.measurements])
    return float(np.interp(t_center, t, f))


def _crlb_sigma(fs: float, n: int, snr_db: float, snr_band_db: float | None = None,
                bandwidth_hz: float | None = None) -> float:
    """1-sigma frequency uncertainty (Hz) from the CRLB.

    For a complex tone the centred phase-slope estimator reaches
    ``var(f) = 6 fs^2 / ((2 pi)^2 SNR N (N^2 - 1))`` with SNR the per-sample
    power ratio (signal power over noise power in the sampling band).  The
    same expression is used for a tone-modulated FM carrier: the phase noise
    that limits the carrier estimate is set by the total noise power, while
    the modulation itself contributes nothing over an integer number of
    modulation periods.  The constant is verified by Monte-Carlo in the C2
    experiments.
    """
    snr = 10.0 ** (snr_db / 10.0)
    denom = (2.0 * math.pi) ** 2 * snr * n * (n * n - 1)
    if denom <= 0:
        return float("inf")
    return float(math.sqrt(6.0 * fs * fs / denom))


EXTRACTORS = {
    "stft": STFTRidgeExtractor,
    "phase": PhaseSlopeExtractor,
    "auto": AutoExtractor,
}


def make_extractor(name: str) -> Extractor:
    if name not in EXTRACTORS:
        raise ValueError(f"unknown extractor {name!r}; choose from {sorted(EXTRACTORS)}")
    return EXTRACTORS[name]()

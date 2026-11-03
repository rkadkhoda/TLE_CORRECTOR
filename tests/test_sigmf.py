import numpy as np

from tlecorrect.sigmfio import SigMFRecording


def test_reads_metadata(golden_out):
    meta = golden_out / "NOAA19_tehran_20261021T181639.sigmf-meta"
    if not meta.exists():
        return
    r = SigMFRecording(meta)
    assert r.datatype == "cf32_le"
    assert r.sample_rate == 24000.0
    assert r.num_samples == 15888000
    assert r.center_freq == 137100000.0
    assert r.start_utc.year == 2026


def test_matches_reference_sigmf_package(golden_out):
    meta = golden_out / "NOAA19_tehran_20261021T181639.sigmf-meta"
    data = golden_out / "NOAA19_tehran_20261021T181639.sigmf-data"
    if not meta.exists() or not data.exists():
        return
    import sigmf

    ref = sigmf.sigmffile.fromfile(str(meta))
    ours = SigMFRecording(meta, data)
    assert int(ref.sample_count) == ours.num_samples
    assert float(ref.get_global_field("core:sample_rate")) == ours.sample_rate
    assert text_datatype(ref) == ours.datatype


def text_datatype(ref):
    return ref.get_global_field("core:datatype")


def test_chunked_read_matches_memmap(golden_out):
    data = golden_out / "NOAA19_tehran_20261021T181639.sigmf-data"
    meta = golden_out / "NOAA19_tehran_20261021T181639.sigmf-meta"
    if not meta.exists():
        return
    r = SigMFRecording(meta, data)
    x = r.read(1000, 4096)
    assert x.dtype == np.complex64
    assert np.allclose(x, np.fromfile(data, dtype=np.complex64, count=4096, offset=1000 * 8))

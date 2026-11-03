import datetime as dt

import pytest

from tlecorrect.tle import (
    TLE,
    datetime_to_epoch,
    epoch_to_datetime,
    parse_tle,
    tle_checksum,
)

L1 = "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999"
L2 = "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912"


def test_checksum_matches_reference_lines():
    assert tle_checksum(L1) == int(L1[68])
    assert tle_checksum(L2) == int(L2[68])


def test_parse_values():
    t = parse_tle(L1 + "\n" + L2)
    assert t.satnum == 33591
    assert t.epoch == dt.datetime(2026, 10, 18, 13, 44, 50, 65152, tzinfo=dt.UTC)
    assert abs(t.n - 14.13490024) < 1e-8
    assert abs(t.i - 98.9402) < 1e-9
    assert abs(t.raan - 2.4602) < 1e-9
    assert abs(t.ecc - 0.0014401) < 1e-12
    assert abs(t.argp - 44.7273) < 1e-9
    assert abs(t.M - 315.5058) < 1e-9
    assert abs(t.bstar - 3.8013e-5) < 1e-12


def test_round_trip_is_byte_identical():
    t = parse_tle(L1 + "\n" + L2)
    assert t.with_elements().line2 == L2
    assert t.with_elements().line1 == L1


def test_element_change_updates_checksum_and_fields():
    t = parse_tle(L1 + "\n" + L2)
    t2 = t.with_elements(n=14.13048503911916, M=320.5880)
    assert t2.line2 != L2
    assert tle_checksum(t2.line2) == int(t2.line2[68])
    assert abs(t2.n - 14.13048503911916) < 1e-8
    assert abs(t2.M - 320.5880) < 1e-9
    # other fields preserved
    assert abs(t2.i - 98.9402) < 1e-9
    assert abs(t2.ecc - 0.0014401) < 1e-12


def test_bad_checksum_raises():
    bad = L1[:-1] + "0" if L1[-1] != "0" else L1[:-1] + "1"
    with pytest.raises(ValueError, match="checksum"):
        parse_tle(bad + "\n" + L2)


def test_missing_line_raises():
    with pytest.raises(ValueError):
        parse_tle(L1)


@pytest.mark.parametrize(
    "dtv",
    [
        dt.datetime(2026, 10, 18, 13, 44, 50, 65152, tzinfo=dt.UTC),
        dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.UTC),
        dt.datetime(2005, 12, 31, 23, 59, 59, 990000, tzinfo=dt.UTC),
    ],
)
def test_epoch_field_round_trip(dtv):
    field = datetime_to_epoch(dtv)
    back = epoch_to_datetime(field)
    assert abs((back - dtv).total_seconds()) < 1e-3


def test_semi_major_axis_reasonable():
    t = parse_tle(L1 + "\n" + L2)
    assert 7.0e6 < t.semi_major_axis_m < 7.4e6
    assert 6000.0 < t.period_s < 6200.0  # seconds (~102 min)


def test_tle_dataclass_requires_epoch():
    with pytest.raises(ValueError):
        TLE(line1=L1, line2=L2)

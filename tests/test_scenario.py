import datetime as dt

import pytest
from pydantic import ValidationError

from tlecorrect.scenario import (
    Config,
    Observation,
    Station,
    load_config_yaml,
    load_scenario_file,
    manifest_to_config,
    save_config_yaml,
    scenario_to_config,
)


def test_scenario_import(golden_out):
    s = load_scenario_file(golden_out / "scenario.json")
    assert s.schema_version == "1.0"
    assert s.stations[0].name == "tehran"
    cfg = scenario_to_config(s, golden_out)
    assert len(cfg.observations) == 1
    o = cfg.observations[0]
    assert o.station_name == "tehran"
    assert o.datatype == "cf32_le"
    assert o.sample_rate_hz == 24000.0
    assert o.fc_hz == 137100000.0
    assert o.flo_hz == 137105000.0
    assert o.num_samples == 15888000
    assert o.start_utc == dt.datetime(2026, 10, 21, 18, 16, 34, tzinfo=dt.UTC)


def test_scenario_conflict_is_reported(golden_out):
    s = load_scenario_file(golden_out / "scenario.json")
    cfg = scenario_to_config(s, golden_out)
    assert any("SDR center frequency differs" in w for w in cfg.warnings)


def test_manifest_and_scenario_agree_on_core_fields(golden_out):
    sc = scenario_to_config(load_scenario_file(golden_out / "scenario.json"), golden_out)
    mf = manifest_to_config(golden_out / "manifest.json")
    assert len(mf.observations) == 1
    a, b = sc.observations[0], mf.observations[0]
    assert a.sample_rate_hz == b.sample_rate_hz
    assert a.datatype == b.datatype
    assert a.start_utc == b.start_utc
    assert a.num_samples == b.num_samples
    assert a.fc_hz == b.fc_hz


def test_missing_required_field_fails():
    with pytest.raises(ValidationError):
        Observation(station_name="x", data_path="a", fc_hz=1, flo_hz=1,
                    datatype="cf32_le", start_utc=dt.datetime.now(dt.UTC))
    with pytest.raises(ValidationError):
        Station(name="x")  # lat/lon required


def test_yaml_manual_equivalence(tmp_path):
    manual = Config(
        stations=[Station(name="tehran", lat_deg=35.7, lon_deg=51.4, alt_m=1200.0,
                          center_offset_hz=5000.0)],
        observations=[
            Observation(id=0, station_name="tehran", data_path="x.sigmf-data",
                        fc_hz=137.1e6, flo_hz=137.105e6, sample_rate_hz=24000.0,
                        datatype="cf32_le",
                        start_utc=dt.datetime(2026, 10, 21, 18, 16, 34, tzinfo=dt.UTC))
        ],
    )
    path = tmp_path / "cfg.yaml"
    save_config_yaml(manual, path)
    loaded = load_config_yaml(path)
    assert loaded.model_dump() == manual.model_dump()


def test_unknown_keys_are_ignored(tmp_path, golden_out):
    import json

    data = json.loads((golden_out / "scenario.json").read_text())
    data["brand_new_key"] = {"x": 1}
    data["stations"][0]["also_new"] = 5
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    s = load_scenario_file(p)
    assert s.stations[0].name == "tehran"


def test_missing_recording_files_raise(tmp_path, golden_out):
    import json

    data = json.loads((golden_out / "scenario.json").read_text())
    data["recordings"][0]["file_names"] = ["does_not_exist.sigmf-data",
                                           "does_not_exist.sigmf-meta"]
    p = tmp_path / "scenario.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    s = load_scenario_file(p)
    from tlecorrect.scenario import ConfigError

    with pytest.raises(ConfigError):
        scenario_to_config(s, tmp_path)

"""Shared pytest fixtures and project paths."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = PROJECT_ROOT / "testdata" / "golden01"
GOLDEN_OUT = GOLDEN / "out"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def golden_dir() -> Path:
    if not GOLDEN.exists():
        pytest.skip("golden01 fixture not present")
    return GOLDEN


@pytest.fixture(scope="session")
def golden_out() -> Path:
    if not GOLDEN_OUT.exists():
        pytest.skip("golden01 fixture not present")
    return GOLDEN_OUT


TRUE_TLE_L1 = "1 33591U 09005A   26291.57280168  .00000027  00000+0  38013-4 0  9999"
TRUE_TLE_L2 = "2 33591  98.9402   2.4602 0014401  44.7273 315.5058 14.13490024911912"

PERT_TLE_L2 = "2 33591  98.9402   2.4525 0014401  44.7273 320.5880 14.13048503911916"


@pytest.fixture(scope="session")
def true_tle_text() -> str:
    return f"{TRUE_TLE_L1}\n{TRUE_TLE_L2}\n"


@pytest.fixture(scope="session")
def pert_tle_text() -> str:
    return f"{TRUE_TLE_L1}\n{PERT_TLE_L2}\n"

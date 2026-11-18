from datetime import UTC, datetime

import pytest

from forecast.data.align import session_align
from forecast.data.synthetic import make_bars
from forecast.features.build import build
from forecast.utils.config import load_yaml


@pytest.fixture(scope="session")
def fcfg():
    return load_yaml("configs/features/v1.yaml")


@pytest.fixture(scope="session")
def bars():
    return make_bars(["SPY", "QQQ", "AAA", "BBB", "CCC"], datetime(2024, 1, 2, tzinfo=UTC),
                     datetime(2024, 1, 20, tzinfo=UTC), seed=1)


@pytest.fixture(scope="session")
def aligned(bars):
    return session_align(bars)


@pytest.fixture(scope="session")
def fs(aligned, fcfg):
    return build(aligned, fcfg)

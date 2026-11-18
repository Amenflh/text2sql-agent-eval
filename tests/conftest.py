import pytest

from t2s.agent import Agent
from t2s.data import build


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    return build(tmp_path_factory.mktemp("db") / "bank.db", seed=42)


@pytest.fixture(scope="session")
def agent(db_path):
    return Agent(db_path, backend="offline")

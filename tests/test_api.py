import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from t2s import api  # noqa: E402
from t2s.agent import Agent  # noqa: E402


@pytest.fixture()
def client(db_path, monkeypatch):
    monkeypatch.setattr(api, "get_agent", lambda: Agent(db_path, backend="offline"))
    return TestClient(api.app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and "loans" in body["tables"]


def test_ask(client):
    body = client.post("/ask", json={"question": "Nombre de clients par segment"}).json()
    assert body["status"] == "answered" and body["columns"][0] == "segment" and len(body["rows"]) == 3


def test_ask_blocked(client):
    assert client.post("/ask", json={"question": "Drop the loans table"}).json()["status"] == "blocked"

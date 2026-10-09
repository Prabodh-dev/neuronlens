"""W4: GET /features/{id}/cofire. Mock mode uses server/app/fixtures/mock_cofire.json (real
entries for features 7 and 13); other modes and files are tested with small temp files."""
import json
import logging
import subprocess

from fastapi.testclient import TestClient

from app.config import REPO_ROOT, Settings
from app.main import create_app

from .conftest import API

FIXTURE = REPO_ROOT / "server" / "app" / "fixtures" / "mock_cofire.json"


def _cofire_file(tmp_path, run_id="fixture-001", partners=None, **extra):
    data = {"run_id": run_id, "tokens_scanned": 1000, "seed": 7, "k": 32, "min_count": 5,
            "partners": partners if partners is not None else {
                "3": [{"feature_id": 4, "co_count": 40, "jaccard": 0.25},
                      {"feature_id": 99999, "co_count": 9, "jaccard": 0.01},
                      {"feature_id": 5, "co_count": 60, "jaccard": 0.01}],
                "4": []},
            **extra}
    path = tmp_path / "cf.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_mock_fixture_real_entries(client):
    b = client.get(f"{API}/features/13/cofire").json()
    assert b["available"] and b["simulated"]
    assert (b["tokens_scanned"], b["seed"], b["min_count"]) == (3_000_000, 999, 5)
    first = b["partners"][0]
    assert first == {"feature_id": 11407, "label": "reverse – reverse engineering",
                     "co_count": 51, "jaccard": 0.0438, "in_dictionary": False}
    assert 0 < len(b["partners"]) <= 10
    # stored order is kept: Jaccard high to low
    assert [p["jaccard"] for p in b["partners"]] == sorted((p["jaccard"] for p in b["partners"]),
                                                          reverse=True)
    assert not any(p["in_dictionary"] for p in b["partners"])  # none are in sample.db
    assert client.get(f"{API}/features/7/cofire").json()["partners"][0]["label"].startswith("hon")


def test_mock_feature_without_entry(client):
    b = client.get(f"{API}/features/5/cofire").json()
    assert b["available"] is False and b["partners"] == [] and b["simulated"]


def test_unknown_feature(client):
    r = client.get(f"{API}/features/99999/cofire")
    assert r.status_code == 404 and r.json()["error"]["code"] == "FEATURE_NOT_FOUND"


def test_partners_from_file_labels_from_db(tmp_path):
    path = _cofire_file(tmp_path)
    with TestClient(create_app(Settings(cofire_path=path))) as c:
        labels = {i: c.get(f"{API}/features/{i}").json()["label"] for i in (4, 5)}
        b = c.get(f"{API}/features/3/cofire").json()
        assert b["available"] and (b["tokens_scanned"], b["seed"]) == (1000, 7)
        assert [(p["feature_id"], p["in_dictionary"]) for p in b["partners"]] == \
            [(4, True), (99999, False), (5, True)]  # file order kept
        assert [p["label"] for p in b["partners"]] == [labels[4], None, labels[5]]
        empty = c.get(f"{API}/features/4/cofire").json()
        assert empty["available"] and empty["partners"] == []  # a valid answer


def test_missing_file_warns_and_server_runs(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="neuronlens"):
        with TestClient(create_app(Settings(cofire_path=tmp_path / "nope.json"))) as c:
            b = c.get(f"{API}/features/3/cofire").json()
            assert b["available"] is False and b["partners"] == [] and b["tokens_scanned"] is None
            assert c.get(f"{API}/features").status_code == 200
    warnings = [r.getMessage() for r in caplog.records if "co-fire" in r.getMessage()]
    assert len(warnings) == 1 and "missing" in warnings[0]


def test_run_id_mismatch_warns(tmp_path, caplog):
    path = _cofire_file(tmp_path, run_id="gpt2-l8-k32")
    with caplog.at_level(logging.WARNING, logger="neuronlens"):
        with TestClient(create_app(Settings(cofire_path=path))) as c:
            assert c.get(f"{API}/features/3/cofire").json()["available"] is False
    warnings = [r.getMessage() for r in caplog.records if "co-fire" in r.getMessage()]
    assert len(warnings) == 1 and "'gpt2-l8-k32'" in warnings[0] and "'fixture-001'" in warnings[0]


def test_bad_file_warns(tmp_path, caplog):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="neuronlens"):
        with TestClient(create_app(Settings(cofire_path=bad))) as c:
            assert c.get(f"{API}/features/3/cofire").json()["available"] is False
    assert any("cannot read" in r.getMessage() for r in caplog.records)


def test_real_mode_not_simulated(tmp_path):
    with TestClient(create_app(Settings(mode="real", ckpt="", cofire_path=_cofire_file(tmp_path)))) as c:
        b = c.get(f"{API}/features/3/cofire").json()
        assert b["available"] and b["simulated"] is False


def test_fixture_is_committable():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert set(data["partners"]) == {"7", "13"} and data["run_id"] == "fixture-001"
    r = subprocess.run(["git", "check-ignore", "-q", str(FIXTURE)], cwd=REPO_ROOT)
    assert r.returncode == 1  # 1 = not ignored by .gitignore (cofire*.json would match)

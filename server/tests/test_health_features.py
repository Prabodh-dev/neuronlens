from .conftest import API


def test_health(client):
    r = client.get(f"{API}/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "ok", "mode": "mock", "is_synthetic": True,
                    "schema_version": "1.0", "model_name": "gpt2", "hook_layer": 8,
                    "d_model": 768, "d_sae": 512, "k": 32, "run_id": "fixture-001"}


def test_list_defaults_hide_dead(client):
    body = client.get(f"{API}/features").json()
    assert body["total"] == 490 and body["limit"] == 50 and body["offset"] == 0
    assert len(body["items"]) == 50
    assert not any(i["is_dead"] for i in body["items"])
    assert [i["feature_id"] for i in body["items"]] == sorted(i["feature_id"] for i in body["items"])


def test_include_dead_and_status(client):
    assert client.get(f"{API}/features?include_dead=true").json()["total"] == 512
    assert client.get(f"{API}/features?status=ambiguous").json()["total"] == 30
    assert client.get(f"{API}/features?status=named").json()["total"] == 400


def test_query_label_and_id(client):
    sports = client.get(f"{API}/features?query=SPORTS&limit=200").json()
    assert sports["total"] > 0 and all("sports" in i["label"].lower() for i in sports["items"])
    by_id = client.get(f"{API}/features?query=%20%2012%20&include_dead=true&limit=200").json()
    assert 12 in [i["feature_id"] for i in by_id["items"]]
    assert client.get(f"{API}/features?query=%20%20").json()["total"] == 490


def test_sort_whitelist_and_pagination_stability(client):
    r = client.get(f"{API}/features?sort=label")
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"
    seen, offset = [], 0
    while True:
        page = client.get(f"{API}/features?sort=max_activation&order=desc&limit=100&offset={offset}").json()
        seen += [i["feature_id"] for i in page["items"]]
        offset += 100
        if offset >= page["total"]:
            break
    assert len(seen) == len(set(seen)) == 490
    acts = {i["feature_id"]: i["max_activation"] for i in
            client.get(f"{API}/features?limit=200").json()["items"]}
    top = client.get(f"{API}/features?sort=max_activation&order=desc&limit=3").json()["items"]
    assert [i["max_activation"] for i in top] == sorted((i["max_activation"] for i in top), reverse=True)
    assert acts  # sanity


def test_limit_bounds(client):
    for qs in ("limit=0", "limit=201", "offset=-1"):
        r = client.get(f"{API}/features?{qs}")
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"


def test_detail(client):
    r = client.get(f"{API}/features/3")
    assert r.status_code == 200
    d = r.json()
    assert d["feature_id"] == 3 and len(d["examples"]) == 5
    assert [e["rank"] for e in d["examples"]] == [1, 2, 3, 4, 5]
    ex = d["examples"][0]
    assert len(ex["tokens"]) == len(ex["activations"])
    assert ex["activations"][ex["peak_index"]] == max(ex["activations"])
    assert any(t.startswith(" ") for e in d["examples"] for t in e["tokens"])
    assert len(d["logits"]["promoted"]) == 5 and len(d["logits"]["suppressed"]) == 5


def test_detail_not_found_and_unknown_route(client):
    r = client.get(f"{API}/features/99999")
    assert r.status_code == 404 and r.json()["error"]["code"] == "FEATURE_NOT_FOUND"
    r = client.get(f"{API}/features/abc")
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"
    r = client.get(f"{API}/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


def _meta_with_version(tmp_path, version):
    import shutil
    import sqlite3
    from pathlib import Path

    from app.db import Database, load_meta
    path = tmp_path / "v.db"
    shutil.copyfile(Path(__file__).resolve().parents[2] / "contracts" / "fixtures" / "sample.db",
                    path)
    con = sqlite3.connect(path)
    with con:
        con.execute("UPDATE meta SET value = ? WHERE key = 'schema_version'", (version,))
    con.close()
    db = Database(path)
    try:
        return load_meta(db)
    finally:
        db.close()


def test_schema_version_is_major_minor(tmp_path):
    import pytest

    from app.errors import ApiError
    # the real DB says "1", the fixture "1.0": both must report "1.0"
    assert _meta_with_version(tmp_path, "1")["schema_version"] == "1.0"
    assert _meta_with_version(tmp_path, " 1 ")["schema_version"] == "1.0"
    assert _meta_with_version(tmp_path, "1.2")["schema_version"] == "1.2"
    for bad in ("2", "2.0", "0.9"):
        with pytest.raises(ApiError):
            _meta_with_version(tmp_path, bad)

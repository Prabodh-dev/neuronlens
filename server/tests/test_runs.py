from .conftest import API


def test_random_feature_is_live_detail(client):
    seen = set()
    for _ in range(10):
        r = client.get(f"{API}/features/random")
        assert r.status_code == 200
        d = r.json()
        assert not d["is_dead"] and len(d["examples"]) == 5 and "logits" in d
        seen.add(d["feature_id"])
    assert len(seen) > 1


def test_steering_examples_curated_first(client):
    items = client.get(f"{API}/steering-examples").json()["items"]
    assert len(items) == 8
    curated = [i["is_curated"] for i in items]
    assert curated == sorted(curated, reverse=True) and curated.count(True) == 4
    assert all(i["mode"] in ("amplify", "ablate") for i in items)


def test_steering_examples_filter(client):
    first = client.get(f"{API}/steering-examples").json()["items"][0]
    items = client.get(f"{API}/steering-examples?feature_id={first['feature_id']}").json()["items"]
    assert items and all(i["feature_id"] == first["feature_id"] for i in items)
    assert client.get(f"{API}/steering-examples?feature_id=99999").json() == {"items": []}


def test_runs(client):
    items = client.get(f"{API}/runs").json()["items"]
    assert items[0]["run_id"] == "fixture-001" and items[0]["is_primary"]
    assert items[0]["hook_layer"] == 8 and items[0]["k"] == 32


def test_metrics_default_run(client):
    body = client.get(f"{API}/metrics").json()
    assert body["run_id"] == "fixture-001"
    names = [m["name"] for m in body["metrics"]]
    assert names == ["fvu", "l0", "dead_pct", "ce_original", "ce_zero_ablation",
                     "ce_reconstructed", "ce_recovered"]
    values = {m["name"]: m["value"] for m in body["metrics"]}
    assert values["fvu"] == 0.121 and values["dead_pct"] == 4.3
    assert len(body["density_hist"]) == 10
    los = [b["bin_lo"] for b in body["density_hist"]]
    assert los == sorted(los)
    steps = [p["step"] for p in body["training_curve"]]
    assert len(steps) == 30 and steps == sorted(steps)


def test_metrics_matches_db(client):
    import sqlite3
    from app.config import Settings
    con = sqlite3.connect(Settings().db_path)
    db = dict(con.execute("SELECT name, value FROM metrics WHERE run_id = 'fixture-001'"))
    con.close()
    api = {m["name"]: m["value"] for m in client.get(f"{API}/metrics").json()["metrics"]}
    assert api == db


def test_metrics_unknown_run(client):
    r = client.get(f"{API}/metrics?run_id=nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "RUN_NOT_FOUND"
    assert client.get(f"{API}/metrics?run_id=fixture-001").status_code == 200

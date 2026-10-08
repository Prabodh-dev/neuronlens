from .conftest import API


def live_feature(client, status="named"):
    return client.get(f"{API}/features?status={status}&limit=1").json()["items"][0]


def dead_feature_id(client):
    items = client.get(f"{API}/features?include_dead=true&limit=200").json()["items"]
    return next(i["feature_id"] for i in items if i["is_dead"])


def test_inspect_shape_and_outlier(client):
    r = client.post(f"{API}/inspect", json={"text": "The court ruled on the case.", "top_n": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "gpt2"
    toks = body["tokens"]
    assert [t["index"] for t in toks] == list(range(len(toks)))
    assert "".join(t["text"] for t in toks) == "The court ruled on the case."
    assert toks[0]["is_outlier_position"] and toks[0]["features"] == []
    for t in toks[1:]:
        assert not t["is_outlier_position"]
        assert 1 <= len(t["features"]) <= 3
        acts = [f["activation"] for f in t["features"]]
        assert acts == sorted(acts, reverse=True) and all(a > 0 for a in acts)
    labelled = [f for t in toks for f in t["features"] if f["label"]]
    assert labelled, "labels are joined from the DB"


def test_inspect_deterministic(client):
    body = {"text": "Music is playing in the hall.", "top_n": 5}
    assert client.post(f"{API}/inspect", json=body).json() == \
        client.post(f"{API}/inspect", json=body).json()


def test_inspect_features_are_live(client):
    toks = client.post(f"{API}/inspect", json={"text": "a b c d e f", "top_n": 20}).json()["tokens"]
    ids = {f["feature_id"] for t in toks for f in t["features"]}
    for fid in ids:
        assert not client.get(f"{API}/features/{fid}").json()["is_dead"]


def test_inspect_validation(client):
    for body in ({"text": "   "}, {"text": "hi", "top_n": 0}, {"text": "hi", "top_n": 21}, {}):
        r = client.post(f"{API}/inspect", json=body)
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"


def test_inspect_too_long(client):
    r = client.post(f"{API}/inspect", json={"text": " word" * 257})
    assert r.status_code == 413 and r.json()["error"]["code"] == "TEXT_TOO_LONG"
    assert client.post(f"{API}/inspect", json={"text": " word" * 256}).status_code == 200


def test_steer_amplify(client):
    f = live_feature(client)
    req = {"prompt": "The story begins", "feature_id": f["feature_id"], "mode": "amplify"}
    r = client.post(f"{API}/steer", json=req)
    assert r.status_code == 200
    body = r.json()
    assert body["feature"] == {"feature_id": f["feature_id"], "label": f["label"]}
    assert body["mode"] == "amplify" and body["strength"] == 1.0 and body["seed"] == 42
    assert body["baseline_text"].startswith("The story begins")
    assert body["steered_text"].startswith("The story begins")
    assert "[simulated]" in body["baseline_text"] and "[simulated]" in body["steered_text"]
    assert body["baseline_text"] != body["steered_text"]


def test_steer_identical_requests_identical_output(client):
    f = live_feature(client)
    req = {"prompt": "Today", "feature_id": f["feature_id"], "mode": "amplify",
           "strength": 1.5, "seed": 7, "max_new_tokens": 30}
    assert client.post(f"{API}/steer", json=req).json() == \
        client.post(f"{API}/steer", json=req).json()


def test_steer_same_seed_same_baseline(client):
    a, b = client.get(f"{API}/features?status=named&limit=2").json()["items"]
    base = {"prompt": "Hello", "mode": "amplify", "seed": 3}
    ra = client.post(f"{API}/steer", json={**base, "feature_id": a["feature_id"]}).json()
    rb = client.post(f"{API}/steer", json={**base, "feature_id": b["feature_id"],
                                           "strength": 2.0}).json()
    assert ra["baseline_text"] == rb["baseline_text"]


def test_steer_zero_strength_matches_baseline(client):
    f = live_feature(client)
    body = client.post(f"{API}/steer", json={"prompt": "Hi", "feature_id": f["feature_id"],
                                             "mode": "amplify", "strength": 0}).json()
    assert body["baseline_text"] == body["steered_text"]


def test_steer_ablate_ignores_strength(client):
    f = live_feature(client)
    r = client.post(f"{API}/steer", json={"prompt": "Hi", "feature_id": f["feature_id"],
                                          "mode": "ablate", "strength": 2.5})
    assert r.status_code == 200 and r.json()["strength"] == 0.0


def test_steer_strength_bounds(client):
    fid = live_feature(client)["feature_id"]
    for s in (-0.1, 3.01):
        r = client.post(f"{API}/steer", json={"prompt": "Hi", "feature_id": fid,
                                              "mode": "amplify", "strength": s})
        assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_INPUT"
    r = client.post(f"{API}/steer", json={"prompt": "Hi", "feature_id": fid,
                                          "mode": "amplify", "strength": 3})
    assert r.status_code == 200


def test_steer_errors(client):
    cases = [
        ({"prompt": "Hi", "feature_id": 99999, "mode": "amplify"}, 404, "FEATURE_NOT_FOUND"),
        ({"prompt": "Hi", "feature_id": dead_feature_id(client), "mode": "amplify"},
         422, "INVALID_INPUT"),
        ({"prompt": " ", "feature_id": 1, "mode": "amplify"}, 422, "INVALID_INPUT"),
        ({"prompt": "Hi", "feature_id": 1, "mode": "boost"}, 422, "INVALID_INPUT"),
        ({"prompt": "Hi", "feature_id": 1, "mode": "amplify", "max_new_tokens": 0},
         422, "INVALID_INPUT"),
        ({"prompt": " word" * 257, "feature_id": live_feature(client)["feature_id"],
          "mode": "amplify"}, 413, "TEXT_TOO_LONG"),
    ]
    for req, status, code in cases:
        r = client.post(f"{API}/steer", json=req)
        assert r.status_code == status and r.json()["error"]["code"] == code, req

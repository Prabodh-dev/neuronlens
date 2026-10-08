import argparse
import json
import sqlite3


def card(c, f):
    d, mx, lab = c.execute(
        "select density,max_activation,label from features where feature_id=?", (f,)).fetchone()
    print(f"=== feature {f}  density {d:.5f}  max {mx:.1f}  label: {lab}")
    for t, p in c.execute(
            "select tokens_json,peak_index from feature_examples "
            "where feature_id=? and rank<=3 order by rank", (f,)):
        t = json.loads(t)
        s = "..." + "".join(t[max(0, p - 10):p]) + "[[" + t[p] + "]]" + "".join(t[p + 1:p + 4])
        print("  " + s.replace("\n", "\\n"))
    for d_, name in (("promoted", "promotes"), ("suppressed", "suppresses")):
        toks = [r[0] for r in c.execute(
            "select token from feature_logits where feature_id=? and direction=? "
            "and rank<=6 order by rank", (f, d_))]
        print(f"  {name}: {toks}")


def by_token(c, tok):
    tok = tok.lower()
    ids = {r[0] for r in c.execute(
        "select feature_id from feature_logits where direction='promoted' "
        "and rank<=5 and lower(token) like ?", (f"%{tok}%",))}
    for f, t, p in c.execute(
            "select feature_id,tokens_json,peak_index from feature_examples where rank=1"):
        if tok in json.loads(t)[p].lower():
            ids.add(f)
    return sorted(ids)


def apply_labels(c, path):
    n = 0
    for k, v in json.load(open(path)).items():
        label, status = (v, "named") if isinstance(v, str) else (v["label"], v.get("status", "named"))
        c.execute("update features set label=?, label_status=? where feature_id=?",
                  (label, status, int(k)))
        n += 1
    c.commit()
    print("labels applied:", n)


p = argparse.ArgumentParser()
p.add_argument("--db", default="../data/neuronlens.db")
p.add_argument("--ids", type=int, nargs="*")
p.add_argument("--token")
p.add_argument("--random", type=int)
p.add_argument("--top", type=int, help="strongest features with moderate density")
p.add_argument("--skip", type=int, default=0)
p.add_argument("--limit", type=int, default=10)
p.add_argument("--apply-labels")
a = p.parse_args()

c = sqlite3.connect(a.db)
if a.apply_labels:
    apply_labels(c, a.apply_labels)
ids = a.ids or []
if a.token:
    found = by_token(c, a.token)
    print(f"{len(found)} features match '{a.token}'; showing {a.limit}")
    ids = found[:a.limit]
if a.random:
    import random
    pool = [r[0] for r in c.execute(
        "select feature_id from features where is_dead=0 and density between 0.0001 and 0.01")]
    ids = random.sample(pool, a.random)
if a.top:
    ids = [r[0] for r in c.execute(
        "select feature_id from features where is_dead=0 and density between 0.0002 and 0.005 "
        "order by max_activation desc limit ? offset ?", (a.top, a.skip))]
for f in ids:
    card(c, f)
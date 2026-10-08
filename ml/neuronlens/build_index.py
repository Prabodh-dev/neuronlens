import datetime
import itertools
import json
import os
import sqlite3

import numpy as np

from .activations import Collector, load_model, token_sequences
from .interpret import build_examples, load_sae, logit_attribution, scan

METRIC_NAMES = {"fvu": "fvu", "l0": "l0", "dead_pct": "dead_pct",
                "ce_orig": "ce_original", "ce_zero": "ce_zero_ablation",
                "ce_recon": "ce_reconstructed", "ce_recovered": "ce_recovered"}


def _training_curve(run_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        ea.Reload()
        mse = {e.step: e.value for e in ea.Scalars("train/mse")}
        dead = {e.step: e.value for e in ea.Scalars("train/dead_pct_train")}
        fvu = {e.step: e.value for e in ea.Scalars("eval/fvu")}
        return [(s, mse[s], fvu.get(s), dead.get(s)) for s in sorted(mse)]
    except Exception as e:
        print("training curve skipped:", e)
        return []


def build(ckpt_path, out_db, schema_path, n_seqs=3000, labels_path="labels.json",
          data_dir="~/data/fineweb/sample/10BT", device="cuda"):
    ckpt_path = os.path.expanduser(ckpt_path)
    sae, ck = load_sae(ckpt_path, device)
    model, tok = load_model(device)
    col = Collector(model, ck["layer"])
    seqs = list(itertools.islice(token_sequences(tok, data_dir, seed=123), n_seqs))
    print("sequences:", len(seqs))

    S = scan(model, col, sae, ck, seqs, device)
    ex = build_examples(model, col, tok, sae, ck, seqs, S, device)
    logits = logit_attribution(model, tok, sae)

    labels = {}
    if os.path.exists(labels_path):
        labels = {str(k): v for k, v in json.load(open(labels_path)).items()}

    F, n_tok = sae.d_sae, S["n_tok"]
    fire, asum = S["fire"], S["asum"]
    density = fire / n_tok
    dead_pct = 100.0 * float((fire == 0).mean())
    run_id = ck["cfg"]["run_id"]
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    if os.path.exists(out_db):
        os.remove(out_db)
    os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
    con = sqlite3.connect(out_db)
    con.executescript(open(schema_path).read())

    meta = {"schema_version": "1", "run_id": run_id, "model_name": "gpt2",
            "hook_layer": ck["layer"], "d_model": ck["d_model"], "d_sae": F,
            "k": ck["k"], "created_at": now, "is_synthetic": "false"}
    con.executemany("INSERT INTO meta VALUES (?,?)", [(k, str(v)) for k, v in meta.items()])
    con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (run_id, "gpt2", ck["layer"], ck["d_model"], F, ck["k"],
                 ck["cfg"].get("n_train_activations"), os.path.basename(ckpt_path), now, None))

    rows = []
    for f in range(F):
        live = fire[f] > 0
        lab = labels.get(str(f))
        rows.append((f, lab, "named" if lab else "unlabelled", float(density[f]),
                     float(S["top_v"][0, f]) if live else 0.0,
                     float(asum[f] / fire[f]) if live else 0.0, 0 if live else 1))
    con.executemany("INSERT INTO features (feature_id,label,label_status,density,"
                    "max_activation,mean_activation,is_dead) VALUES (?,?,?,?,?,?,?)", rows)

    con.executemany(
        "INSERT INTO feature_examples (feature_id,rank,tokens_json,activations_json,"
        "peak_index,max_activation) VALUES (?,?,?,?,?,?)",
        [(f, i + 1, json.dumps(t, ensure_ascii=False), json.dumps(a), pk, mx)
         for f, lst in ex.items() for i, (r, t, a, pk, mx) in enumerate(lst)])

    con.executemany(
        "INSERT INTO feature_logits VALUES (?,?,?,?,?)",
        [(f, d, i + 1, tkn, sc) for f, dd in logits.items()
         for d, lst in dd.items() for i, (tkn, sc) in enumerate(lst)])

    m = {METRIC_NAMES[k]: float(v) for k, v in ck["metrics"].items() if k in METRIC_NAMES}
    m["dead_pct"] = dead_pct
    con.executemany("INSERT INTO metrics VALUES (?,?,?)", [(run_id, k, v) for k, v in m.items()])

    live_d = density[density > 0]
    counts, edges = np.histogram(np.log10(live_d), bins=np.arange(-7, 0.01, 0.25))
    con.executemany("INSERT INTO density_hist VALUES (?,?,?,?)",
                    [(run_id, float(edges[i]), float(edges[i + 1]), int(counts[i]))
                     for i in range(len(counts))])

    curve = _training_curve(os.path.dirname(ckpt_path))
    con.executemany("INSERT INTO training_curve VALUES (?,?,?,?,?)",
                    [(run_id, s, l, fv, dp) for s, l, fv, dp in curve])
    con.commit()
    con.close()
    print(f"wrote {out_db}: {F} features, {int((fire == 0).sum())} dead in sample, "
          f"{sum(len(v) for v in ex.values())} examples")
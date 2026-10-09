"""Read-only review sheet for trial feature labels.

    cd ml
    PYTHONPATH=. python scripts/review_labels.py --db ../data/neuronlens.db \
        --labels <trial_labels.json|.jsonl> --start 6 --end 19 --examples 6 [--out sheet.txt]
    PYTHONPATH=. python scripts/review_labels.py --db ../data/neuronlens.db \
        --ids-file ../data/spotcheck.csv          # exactly those ids; labels from the database

--ids-file is a CSV with a feature id column (header feature_id, id or feature, as in the
spot-check sheet written by auto_label.py) or a plain list of ids, one per line. It replaces
--start/--end. Without --labels the label, label_status and notes come from the features
table of the database itself.

For each feature id in the range it prints the proposed label (with type and status when the
labels file has them), the top examples by activation with the peak token wrapped in [[ ]],
and the top promoted tokens from feature_logits. The database is opened read-only
(sqlite3 URI mode=ro) and never written. No network access.

Labels file formats accepted (the trial file's real format was not available when this was
written, so these are the shapes it is expected to take):
  * JSON object {"6": "label text", "7": {"label": "...", "type": "...", "status": "..."}}
  * JSON list [{"feature_id": 6, "label": "...", "type": "...", "status": "..."}, ...]
  * JSONL, one such object per line
  * either of the first two wrapped in a single key such as {"labels": ...}
Id keys: feature_id | id | feature. Label keys: label | proposed_label | name | text.
Type keys: type | label_type | kind. Status keys: status | label_status.
"""
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

ID_KEYS = ("feature_id", "id", "feature")
LABEL_KEYS = ("label", "proposed_label", "name", "text")
TYPE_KEYS = ("type", "label_type", "kind")
STATUS_KEYS = ("status", "label_status")
CONTEXT = 12  # tokens shown on each side of the peak
TOP_LOGITS = 8


def _first(d, keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def _entry(fid, value):
    """Normalise one labels-file entry to {"label", "type", "status"}."""
    if isinstance(value, str):
        return {"label": value, "type": None, "status": None}
    if isinstance(value, dict):
        return {"label": _first(value, LABEL_KEYS), "type": _first(value, TYPE_KEYS),
                "status": _first(value, STATUS_KEYS)}
    raise ValueError(f"feature {fid}: entry must be a string or an object, got {type(value).__name__}")


def load_labels(path):
    """Returns {feature_id (int): {"label", "type", "status"}}."""
    text = Path(path).expanduser().read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:  # JSONL: one JSON value per line
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict) and len(data) == 1:
        (key, only), = data.items()
        if isinstance(only, (list, dict)) and not str(key).strip().lstrip("-").isdigit():
            data = only  # wrapper such as {"labels": [...]}; a lone {"6": {...}} is an id
    out = {}
    if isinstance(data, dict):
        if any(k in data for k in ID_KEYS) and any(k in data for k in LABEL_KEYS):
            data = [data]  # a single object
        else:
            for k, v in data.items():
                out[int(k)] = _entry(k, v)
            return out
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict) or _first(item, ID_KEYS) is None:
                raise ValueError(f"list entry without a feature id: {item!r}")
            fid = int(_first(item, ID_KEYS))
            out[fid] = _entry(fid, item)
        return out
    raise ValueError("labels file must hold a JSON object, a list of objects, or JSONL")


def load_ids(path):
    """Feature ids from a CSV with an id column (header feature_id | id | feature) or a plain
    list of ids, one per line. Order kept, duplicates dropped."""
    with open(Path(path).expanduser(), encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"ids file is empty: {path}")
    first = [c.strip().lower() for c in rows[0]]
    col = next((first.index(k) for k in ID_KEYS if k in first), None)
    if col is not None:
        body, row_no = rows[1:], 2
    elif first[0].lstrip("-").isdigit():
        col, body, row_no = 0, rows, 1  # plain list, no header
    else:
        raise ValueError(f"no feature id column in {path}: header is {rows[0][:6]!r} "
                         "(expected feature_id, id or feature)")
    ids = []
    for n, r in enumerate(body, row_no):
        cell = r[col].strip() if col < len(r) else ""
        if not cell.lstrip("-").isdigit():
            raise ValueError(f"row {n} of {path}: not a feature id: {cell!r}")
        if int(cell) not in ids:
            ids.append(int(cell))
    return ids


def load_db_labels(con, ids):
    """label, label_status and notes from the features table (contracts/schema.sql)."""
    out = {}
    for fid in ids:
        row = con.execute("SELECT label, label_status, notes FROM features WHERE feature_id=?",
                          (fid,)).fetchone()
        if row is not None:
            out[fid] = {"label": row[0], "type": None, "status": row[1],
                        "notes": " ".join(row[2].split()) if row[2] else None}
    return out


def open_readonly(db_path):
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def show(piece):
    """Web-style visible whitespace (web/src/lib/format.ts); empty pieces stay empty."""
    return piece.replace("\r", "").replace("\n", "↵").replace("\t", "⇥")


def mark_example(tokens, acts, peak_hint=None, context=CONTEXT):
    """Context window around the peak token with the peak wrapped in [[ ]].

    The peak is the argmax of the stored token activations (peak_hint is used only if the
    activations are unusable). If the peak piece is empty (the first byte-pieces of a split
    character), the mark spans on to the piece that completes the character.
    Returns (text, peak_activation).
    """
    if not tokens:
        return "", 0.0
    if acts and len(acts) == len(tokens):
        peak = max(range(len(acts)), key=lambda i: acts[i])
        value = float(acts[peak])
    else:
        peak = min(max(int(peak_hint or 0), 0), len(tokens) - 1)
        value = float(acts[peak]) if acts and peak < len(acts) else 0.0
    lo, hi = max(0, peak - context), min(len(tokens), peak + context + 1)
    end = peak
    if tokens[peak] == "":
        nxt = next((i for i in range(peak + 1, hi) if tokens[i] != ""), None)
        end = nxt if nxt is not None else peak
    before = "".join(show(t) for t in tokens[lo:peak])
    marked = "".join(show(t) for t in tokens[peak:end + 1])
    after = "".join(show(t) for t in tokens[end + 1:max(hi, end + 1)])
    return f"{before}[[{marked}]]{after}", value


def _has_table(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (name,)).fetchone() is not None


def build_report(con, labels, start, end, n_examples, source="labels file"):
    return build_report_ids(con, labels, range(start, end + 1), n_examples, source)


def build_report_ids(con, labels, ids, n_examples, source="labels file"):
    lines = []
    has_logits = _has_table(con, "feature_logits")
    for fid in ids:
        entry = labels.get(fid)
        head = f"Feature {fid}"
        if entry is None:
            head += f" | proposed label: (none in {source})"
        else:
            head += f" | proposed label: {entry['label'] if entry['label'] else '(empty)'}"
            if entry["type"]:
                head += f" | type: {entry['type']}"
            if entry["status"]:
                head += f" | status: {entry['status']}"
            if entry.get("notes"):
                head += f" | notes: {entry['notes']}"
        lines.append(head)
        if con.execute("SELECT 1 FROM features WHERE feature_id=?", (fid,)).fetchone() is None:
            lines += ["  (feature not in database)", ""]
            continue
        rows = con.execute(
            "SELECT tokens_json, activations_json, peak_index, max_activation "
            "FROM feature_examples WHERE feature_id=? "
            "ORDER BY max_activation DESC, rank LIMIT ?", (fid, n_examples)).fetchall()
        if not rows:
            lines.append("  (no examples)")
        for i, (tj, aj, pk, mx) in enumerate(rows, 1):
            text, value = mark_example(json.loads(tj), json.loads(aj), pk)
            lines.append(f"  {i}. [{value:.2f}] {text}")
        if has_logits:
            top = con.execute(
                "SELECT token, score FROM feature_logits WHERE feature_id=? "
                "AND direction='promoted' ORDER BY rank LIMIT ?", (fid, TOP_LOGITS)).fetchall()
            if top:
                lines.append("  promoted: " + ", ".join(
                    f"{json.dumps(t, ensure_ascii=False)} ({s:.2f})" for t, s in top))
        lines.append("")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="../data/neuronlens.db")
    p.add_argument("--labels", default=None,
                   help="JSON or JSONL file of trial labels; without it the labels come "
                        "from the database features table")
    p.add_argument("--ids-file", default=None,
                   help="CSV with a feature id column, or one id per line; replaces --start/--end")
    p.add_argument("--start", type=int, default=6)
    p.add_argument("--end", type=int, default=19)
    p.add_argument("--examples", type=int, default=6)
    p.add_argument("--out", default=None, help="also write the text output to this file")
    a = p.parse_args(argv)
    if (not a.ids_file and a.start > a.end) or a.examples < 1:
        p.error("need --start <= --end and --examples >= 1")
    try:
        ids = load_ids(a.ids_file) if a.ids_file else list(range(a.start, a.end + 1))
        labels = load_labels(a.labels) if a.labels else None
        con = open_readonly(a.db)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    try:
        source = "labels file"
        if labels is None:
            labels, source = load_db_labels(con, ids), "database"
        lines = build_report_ids(con, labels, ids, a.examples, source)
    finally:
        con.close()
    text = "\n".join(lines).rstrip("\n") + "\n"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(text)
    if a.out:
        Path(a.out).expanduser().write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

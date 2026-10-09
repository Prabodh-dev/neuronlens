"""Read-only check of the labels in a NeuronLens database against the label standard.

    cd ml
    PYTHONPATH=. python scripts/check_labels.py --db ../data/neuronlens.db [--max-list 50]

The database is opened read-only (sqlite3 URI mode=ro) and never written.

Standard checked (the same rules auto_label.py applies): at most 32 characters, lowercase,
no quote marks, no parentheses, no ending punctuation. A word-piece label such as "hon-" may
end in a hyphen, so a hyphen is not treated as punctuation.

Report: counts of named / ambiguous / unlabelled features; labels that break each rule; the
15 most repeated labels; the ids of every unlabelled feature that is not dead.
Exit code 1 if any label breaks the standard, otherwise 0.
"""
import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

MAX_LABEL_LEN = 32
QUOTES = "\"'`‘’“”«»"
PARENS = "()"
END_PUNCT = ".,;:!?…"
TOP_REPEATED = 15


def open_readonly(db_path):
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    con = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    con.execute("PRAGMA query_only = ON")
    return con


def problems(label):
    """The rules a label breaks, as short names."""
    out = []
    if len(label) > MAX_LABEL_LEN:
        out.append("too long")
    if any(c in QUOTES for c in label):
        out.append("quote mark")
    if any(c in PARENS for c in label):
        out.append("parenthesis")
    if label.rstrip() and label.rstrip()[-1] in END_PUNCT:
        out.append("ends in punctuation")
    if label != label.lower():
        out.append("not lowercase")
    return out


def analyse(con):
    rows = con.execute("SELECT feature_id, label, label_status, is_dead FROM features "
                       "ORDER BY feature_id").fetchall()
    status = Counter(r[2] for r in rows)
    labelled = [(fid, label) for fid, label, _, _ in rows if label]
    by_rule = {}
    for fid, label in labelled:
        for rule in problems(label):
            by_rule.setdefault(rule, []).append((fid, label))
    return {
        "total": len(rows),
        "status": {k: status.get(k, 0) for k in ("named", "ambiguous", "unlabelled")},
        "labelled": len(labelled),
        "by_rule": by_rule,
        "breaking": len({fid for v in by_rule.values() for fid, _ in v}),
        "repeated": Counter(label for _, label in labelled).most_common(TOP_REPEATED),
        "unlabelled_live": [r[0] for r in rows if r[2] == "unlabelled" and not r[3]],
        "status_without_label": [r[0] for r in rows if r[2] in ("named", "ambiguous") and not r[1]],
    }


def _wrap_ids(ids, width=100):
    lines, cur = [], ""
    for i in map(str, ids):
        if cur and len(cur) + 2 + len(i) > width:
            lines.append(cur)
            cur = i
        else:
            cur = f"{cur}, {i}" if cur else i
    return lines + ([cur] if cur else [])


def format_report(r, max_list=50):
    out = [f"features: {r['total']}",
           f"named: {r['status']['named']}",
           f"ambiguous: {r['status']['ambiguous']}",
           f"unlabelled: {r['status']['unlabelled']}",
           "", f"label standard: at most {MAX_LABEL_LEN} characters, lowercase, no quote marks, "
           "no parentheses, no ending punctuation"]
    for rule in ("too long", "quote mark", "parenthesis", "ends in punctuation", "not lowercase"):
        items = r["by_rule"].get(rule, [])
        out.append(f"{rule}: {len(items)}")
        for fid, label in items[:max_list]:
            out.append(f"  {fid}: {label}")
        if len(items) > max_list:
            out.append(f"  ... and {len(items) - max_list} more (--max-list)")
    out.append(f"labels breaking the standard: {r['breaking']}")
    if r["status_without_label"]:
        out.append(f"named/ambiguous with no label text: {len(r['status_without_label'])} "
                   f"(ids: {', '.join(map(str, r['status_without_label'][:20]))})")
    out += ["", f"most repeated labels (top {TOP_REPEATED}):"]
    repeated = [(label, n) for label, n in r["repeated"] if n > 1]
    out += [f"  {n} x {label}" for label, n in repeated] or ["  (no label is used more than once)"]
    out += ["", f"unlabelled features that are not dead: {len(r['unlabelled_live'])}"]
    out += [f"  {line}" for line in _wrap_ids(r["unlabelled_live"])]
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default="../data/neuronlens.db")
    p.add_argument("--max-list", type=int, default=50,
                   help="most labels to list per broken rule (counts are always complete)")
    a = p.parse_args(argv)
    try:
        con = open_readonly(a.db)
    except OSError as e:
        sys.exit(f"error: {e}")
    try:
        r = analyse(con)
    finally:
        con.close()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(format_report(r, a.max_list))
    return 1 if r["breaking"] else 0


if __name__ == "__main__":
    sys.exit(main())

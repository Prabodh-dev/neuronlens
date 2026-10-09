#!/usr/bin/env python3
"""Real-mode check for W8 (POST /features/search): prints the top features for a few
queries, each with and without a leading space on the first word.

Run from the repo root on a machine with the checkpoint:
    NEURONLENS_MODE=real NEURONLENS_DB=data/neuronlens.db NEURONLENS_CKPT=data/gpt2-l8-k32.pt \
        python server/scripts/w8_check.py ["another query" ...]
"""
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

QUERIES = ["music", "court ruling", "recipe for pasta"]


def main() -> int:
    settings = Settings.from_env()
    if settings.mode != "real":
        print("set NEURONLENS_MODE=real (and NEURONLENS_DB, NEURONLENS_CKPT)")
        return 1
    queries = sys.argv[1:] or QUERIES
    with TestClient(create_app(settings)) as c:
        for q in queries:
            for query in (q, " " + q.lstrip()):
                r = c.post("/api/v1/features/search", json={"query": query, "top_n": 10})
                if r.status_code != 200:
                    print(f"{query!r}: {r.status_code} {r.json()}")
                    return 1
                b = r.json()
                print(f"\n=== {query!r}  tokens={b['tokens']}")
                for f in b["features"]:
                    print(f"  {f['feature_id']:>5}  {f['max_activation']:6.2f}  "
                          f"rel {f['relative_to_max'] or 0:4.2f}  dens {f['density']:.4f}  "
                          f"tok {b['tokens'][f['peak_token']]!r:12}  {f['label']}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())

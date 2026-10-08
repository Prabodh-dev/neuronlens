#!/usr/bin/env python3
"""Write web/src/lib/api/pending.openapi.json from the server's Pydantic models.

The C3, C4 and C5 endpoints (inspect, steer, steering examples, runs, metrics) have
models in server/app/models.py but are not routed yet, so they are missing from
contracts/openapi.json. This exports just those models so the web app can generate
types from them instead of hand-writing them. Delete this script, the JSON it writes and
pending.d.ts once those schemas appear in contracts/openapi.json.

Run from the repo root (needs server/requirements.txt installed):
    python web/scripts/gen_pending_types.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

from pydantic.json_schema import models_json_schema  # noqa: E402

from app import models  # noqa: E402

PENDING = [
    models.InspectRequest, models.InspectResponse,
    models.SteerRequest, models.SteerResponse,
    models.SteeringExampleList,
    models.RunList,
    models.MetricsResponse,
]


def main() -> None:
    _, schema = models_json_schema([(m, "validation") for m in PENDING],
                                   ref_template="#/components/schemas/{model}")
    spec = {"openapi": "3.1.0", "info": {"title": "NeuronLens pending models", "version": "0"},
            "paths": {}, "components": {"schemas": schema.get("$defs", {})}}
    out = ROOT / "web" / "src" / "lib" / "api" / "pending.openapi.json"
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write contracts/openapi.json deterministically. Run: python server/scripts/export_openapi.py"""
import json
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def main() -> None:
    spec = create_app(Settings()).openapi()
    out = SERVER.parent / "contracts" / "openapi.json"
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                   newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real-mode check for W5 (POST /steer/multi): runs the recorded example from
demo/phase4_examples.json through the real server and prints the response, then compares
the activations with the recorded ones.

Run from the repo root on a machine with the checkpoint:
    NEURONLENS_MODE=real NEURONLENS_DB=data/neuronlens.db NEURONLENS_CKPT=data/gpt2-l8-k32.pt \
        python server/scripts/w5_check.py
"""
import json
import sys
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import REPO_ROOT, Settings  # noqa: E402
from app.main import create_app  # noqa: E402


def main() -> int:
    settings = Settings.from_env()
    if settings.mode != "real":
        print("set NEURONLENS_MODE=real (and NEURONLENS_DB, NEURONLENS_CKPT)")
        return 1
    demo = json.loads((REPO_ROOT / "demo" / "phase4_examples.json").read_text(encoding="utf-8"))
    ex = demo["multi"][0]
    body = {"prompt": ex["prompt"], "seed": ex["seed"], "max_new_tokens": ex["max_new_tokens"],
            "steers": [{"feature_id": s["feature_id"], "strength": s["strength"]}
                       for s in ex["steers"]],
            "with_single_controls": True}
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/v1/steer/multi", json=body)
    print(json.dumps(r.json(), indent=1, ensure_ascii=False))
    if r.status_code != 200:
        return 1
    got = r.json()
    pairs = [("combined", a["target_activation_steered"], e["target_activation_steered"])
             for a, e in zip(got["activations"], ex["activations"])]
    pairs += [("alone", a["target_activation_steered"], e["target_activation_steered"])
              for a, e in zip(got["single_feature_runs"], ex["single_feature_runs"])]
    ok = all(round(g, 2) == round(e, 2) for _, g, e in pairs)
    print("\nactivation check (real vs recorded):")
    for name, g, e in pairs:
        print(f"  {name:8} {g:.2f} vs {e:.2f}  {'ok' if round(g, 2) == round(e, 2) else 'DIFFERS'}")
    print("MATCH" if ok else "MISMATCH: do not open the PR")
    return 0 if ok else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())

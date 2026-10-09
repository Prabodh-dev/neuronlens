"""Loads ml/neuronlens/steering_checks.py (the Phase 4 steering checks). The module is pure
Python, so mock mode can use it without torch; both modes run the same check code."""
import sys

from ..config import REPO_ROOT


def steering_checks():
    ml_dir = str(REPO_ROOT / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from neuronlens import steering_checks as sc  # noqa: PLC0415
    return sc

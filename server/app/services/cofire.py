"""W4 co-fire data: for each feature, the features that fired on the same tokens most often
in a token sample (built by ml/scripts/build_cofire.py; fields in docs/phase4-data-files.md).
Loaded once at start. A missing or mismatched file never stops the server: the endpoint
answers available=false, and one warning in the log says why."""
import json
import logging
from pathlib import Path

log = logging.getLogger("neuronlens")


class Cofire:
    def __init__(self, data: dict | None, simulated: bool):
        self.data = data
        self.simulated = simulated

    @property
    def available(self) -> bool:
        return self.data is not None

    def partners(self, feature_id: int) -> list[dict] | None:
        """The stored list (best first), or None when this feature is not in the file."""
        if self.data is None:
            return None
        return self.data["partners"].get(str(feature_id))

    def label(self, feature_id: int) -> tuple[bool, str | None]:
        """(has a stored label, label). Only the mock fixture stores labels."""
        labels = (self.data or {}).get("labels") or {}
        key = str(feature_id)
        return (key in labels, labels.get(key))


def load_cofire(path: Path, run_id: str, simulated: bool) -> Cofire:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning("co-fire data not available: %s is missing (W4 endpoint returns "
                    "available=false)", path)
        return Cofire(None, simulated)
    except (OSError, ValueError) as e:
        log.warning("co-fire data not available: cannot read %s: %s", path, e)
        return Cofire(None, simulated)
    missing = [k for k in ("run_id", "tokens_scanned", "seed", "min_count", "partners")
               if k not in data]
    if missing:
        log.warning("co-fire data not available: %s lacks %s", path, ", ".join(missing))
        return Cofire(None, simulated)
    if data["run_id"] != run_id:
        log.warning("co-fire data not available: %s is for run %r but the database is run %r",
                    path, data["run_id"], run_id)
        return Cofire(None, simulated)
    return Cofire(data, simulated)

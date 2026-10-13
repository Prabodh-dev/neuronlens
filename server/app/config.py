import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = "contracts/fixtures/sample.db"
DEFAULT_COFIRE = {"real": "data/cofire.json",                      # W4, built by ml/scripts/build_cofire.py
                  "mock": "server/app/fixtures/mock_cofire.json"}  # a few real entries


def resolve_path(value: str) -> Path:
    """Absolute paths as-is; relative paths are tried against the cwd, then the repo root."""
    p = Path(os.path.expanduser(value))
    if p.is_absolute():
        return p
    if (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    return (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    mode: str = "mock"
    db_path: Path = field(default_factory=lambda: resolve_path(DEFAULT_DB))
    ckpt: str = ""
    cors_origins: tuple = ("http://localhost:3000",)
    cofire_path: Path | None = None  # None: DEFAULT_COFIRE for the mode

    def cofire_file(self) -> Path:
        return self.cofire_path or resolve_path(DEFAULT_COFIRE[self.mode])

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.environ.get("NEURONLENS_MODE", "mock").strip().lower() or "mock"
        if mode not in ("mock", "real"):
            raise ValueError(f"NEURONLENS_MODE must be 'mock' or 'real', got {mode!r}")
        origins = os.environ.get("NEURONLENS_CORS_ORIGINS", "http://localhost:3000")
        return cls(
            mode=mode,
            db_path=resolve_path(os.environ.get("NEURONLENS_DB") or DEFAULT_DB),
            ckpt=os.environ.get("NEURONLENS_CKPT", "").strip(),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            cofire_path=(resolve_path(os.environ["NEURONLENS_COFIRE"])
                         if os.environ.get("NEURONLENS_COFIRE", "").strip() else None),
        )

"""Real-mode adapter: wraps ml/neuronlens/runtime.py. Imported only when NEURONLENS_MODE=real,
so mock mode never needs torch or transformers."""
import sys
import threading

from ..config import REPO_ROOT, resolve_path


class RealRuntime:
    is_simulated = False

    def __init__(self, runtime):
        self._rt = runtime
        # the model and its hooks are shared state. Reentrant, so a Phase 4 check can hold
        # it across all of its generations (exclusive) while each generate() also takes it.
        self._lock = threading.RLock()

    @classmethod
    def load(cls, ckpt: str) -> "RealRuntime":
        if not ckpt:
            raise RuntimeError("NEURONLENS_CKPT is not set")
        path = resolve_path(ckpt)
        if not path.is_file():
            raise RuntimeError(f"checkpoint not found: {path}")
        ml_dir = str(REPO_ROOT / "ml")
        if ml_dir not in sys.path:
            sys.path.insert(0, ml_dir)
        import torch  # noqa: PLC0415
        from neuronlens.runtime import Runtime  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(Runtime.load(str(path), device=device))

    def count_tokens(self, text: str) -> int:
        return self._rt.count_tokens(text)

    def inspect(self, text: str, top_n: int) -> list[dict]:
        with self._lock:
            return self._rt.inspect(text, top_n)

    def steer(self, prompt, feature_id, mode, strength, max_activation, max_new_tokens, seed):
        with self._lock:
            return self._rt.steer(prompt, feature_id, mode, strength, max_activation,
                                  max_new_tokens, seed)

    # ---- used by the Phase 4 checks (ml/neuronlens/steering_checks.py) ----

    @property
    def d_sae(self) -> int:
        return self._rt.d_sae

    @property
    def model_name(self) -> str:
        return self._rt.model_name

    @property
    def hook_layer(self) -> int:
        return self._rt.hook_layer

    @property
    def d_model(self) -> int:
        return self._rt.d_model

    def generate(self, prompt, max_new_tokens, seed, steering=None):
        with self._lock:
            return self._rt.generate(prompt, max_new_tokens, seed, steering)

    def feature_activations(self, token_ids, feature_id):
        """X1: one feature's activation per position of a finished text, no steering hook."""
        with self._lock:
            return self._rt.feature_activations(token_ids, feature_id)

    def exclusive(self):
        """Hold the model for a whole check (several generations), like one /steer call."""
        return self._lock

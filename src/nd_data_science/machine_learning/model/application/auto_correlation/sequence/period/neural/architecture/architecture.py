# file: nd_data_science/ml/period_estimation/architecture/architecture.py
from typing import Any, Dict


class Architecture:
    def __init__(self, window_length: int = 128, latent_size: int = 16):
        self._window_length = int(window_length)
        self._latent_size = int(latent_size)

        if self._window_length <= 1:
            raise ValueError("window_length must be > 1.")
        if self._latent_size <= 0:
            raise ValueError("latent_size must be > 0.")

    def get_window_length(self) -> int:
        return int(self._window_length)

    def get_latent_size(self) -> int:
        return int(self._latent_size)

    def to_dict(self) -> Dict[str, Any]:
        return {"window_length": int(self._window_length), "latent_size": int(self._latent_size)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Architecture":
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict.")
        return cls(window_length=int(payload["window_length"]), latent_size=int(payload["latent_size"]))

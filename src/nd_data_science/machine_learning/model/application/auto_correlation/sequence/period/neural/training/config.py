# file: nd_data_science/ml/period_estimation/training/config.py
from typing import Any, Dict


class Config:
    def __init__(self, batch_size: int = 256, epochs: int = 5):
        self._batch_size = int(batch_size)
        self._epochs = int(epochs)

        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self._epochs <= 0:
            raise ValueError("epochs must be > 0.")

    def get_batch_size(self) -> int:
        return int(self._batch_size)

    def get_epochs(self) -> int:
        return int(self._epochs)

    def to_dict(self) -> Dict[str, Any]:
        return {"batch_size": int(self._batch_size), "epochs": int(self._epochs)}

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        if not isinstance(payload, dict):
            raise TypeError("payload must be dict.")
        return cls(batch_size=int(payload["batch_size"]), epochs=int(payload["epochs"]))

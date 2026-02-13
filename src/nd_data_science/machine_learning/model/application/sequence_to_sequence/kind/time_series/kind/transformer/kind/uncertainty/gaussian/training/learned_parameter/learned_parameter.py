# file: nd_data_science/machine_learning/model/application/sequence_to_sequence/kind/time_series/kind/transformer/training/learned_parameter.py
from __future__ import annotations

from typing import List, Optional, Dict, Any

import numpy as np


class LearnedParameter:
    def __init__(self, weights: Optional[List] = None):
        self._weights = weights

    def get_weights(self):
        return self._weights

    def to_npz(self) -> Dict[str, Any]:
        if self._weights is None:
            raise ValueError("weights is None.")

        if not isinstance(self._weights, list):
            raise TypeError("weights must be a list.")

        payload: Dict[str, Any] = {
            "count": np.asarray([len(self._weights)], dtype=np.int64)
        }

        for index, weight in enumerate(self._weights):
            payload[f"w{index}"] = np.asarray(weight)

        return payload

    @classmethod
    def from_npz(cls, payload: Dict[str, Any]) -> "LearnedParameter":
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict.")
        if "count" not in payload:
            raise KeyError("payload must contain 'count'.")

        count = int(np.asarray(payload["count"]).ravel()[0])

        weights: List[np.ndarray] = []
        for index in range(count):
            key = f"w{index}"
            if key not in payload:
                raise KeyError(f"missing key: {key}")
            weights.append(np.asarray(payload[key]))

        return cls(weights=weights)

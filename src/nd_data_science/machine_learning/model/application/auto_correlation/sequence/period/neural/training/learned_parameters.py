# file: nd_data_science/ml/period_estimation/training/learned_parameters.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


class LearnedParameters:
    def __init__(self, weights: List[np.ndarray]):
        if not isinstance(weights, list) or len(weights) == 0:
            raise ValueError("weights must be a non-empty list of np.ndarray.")
        for item in weights:
            if not isinstance(item, np.ndarray):
                raise TypeError("each weight must be np.ndarray.")
        self._weights = weights

    def get_weights(self) -> List[np.ndarray]:
        return self._weights

    def save(self, file_path: str) -> None:
        target = Path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {}
        index = 0
        for weight in self._weights:
            payload[f"weight_{index}"] = weight
            index += 1

        np.savez_compressed(str(target), **payload)

    @classmethod
    def load(cls, file_path: str) -> "LearnedParameters":
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(str(source))

        data = np.load(str(source), allow_pickle=False)

        weights: List[np.ndarray] = []
        index = 0
        while True:
            key = f"weight_{index}"
            if key not in data:
                break
            weights.append(np.array(data[key]))
            index += 1

        if len(weights) == 0:
            raise RuntimeError("No weights found in the file.")

        return cls(weights=weights)

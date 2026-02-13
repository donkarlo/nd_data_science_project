from pathlib import Path
from typing import Dict, Any

import numpy as np

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.learned_parameter.learned_parameter import \
    LearnedParameter


class Storage:
    """Save/load LearnedParameter as NPZ."""

    def save(self, learned_parameter: LearnedParameter, file_path: Path) -> None:
        if not isinstance(learned_parameter, LearnedParameter):
            raise TypeError("learned_parameter must be LearnedParameter.")
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be Path.")

        payload = learned_parameter.to_npz()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(file_path), **payload)

    def load(self, file_path: Path) -> LearnedParameter:
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be Path.")
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        data = np.load(str(file_path), allow_pickle=False)

        payload: Dict[str, Any] = {}
        for key in data.files:
            payload[key] = data[key]

        return LearnedParameter.from_npz(payload)

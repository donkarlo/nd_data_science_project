# file: nd_data_science/machine_learning/model/application/sequence_to_sequence/kind/time_series/kind/transformer/kind/uncertainty/gaussian/architecture_yaml_repository.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.architecture.architecture import  Architecture


class Storage:
    """Save/load Gaussian Architecture as YAML."""

    def save(self, architecture: Architecture, file_path: Path) -> None:
        if not isinstance(architecture, Architecture):
            raise TypeError("config must be Architecture.")
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be Path.")

        payload = architecture.to_dict()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, file_path: Path) -> Architecture:
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be Path.")
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        return self.from_dict(payload)

    def from_dict(self, payload: Dict[str, Any]) -> Architecture:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict.")

        return Architecture.from_dict(payload)

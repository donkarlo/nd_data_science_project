from pathlib import Path
from typing import Dict, Any

import yaml

from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series.kind.transformer.kind.uncertainty.gaussian.training.config import \
    Config


class Storaged:
    """Save/load Training Config as YAML."""

    def save(self, config: Config, file_path: Path) -> None:
        if not isinstance(config, Config):
            raise TypeError("config must be Config.")
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be Path.")

        payload = config.to_dict()

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            yaml.safe_dump(payload, sort_keys=True),
            encoding="utf-8",
        )

    def load(self, file_path: Path) -> Config:
        if not isinstance(file_path, Path):
            raise TypeError("file_path must be Path.")
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        return self.from_dict(payload)

    def from_dict(self, payload: Dict[str, Any]) -> Config:
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict.")

        return Config.from_dict(payload)
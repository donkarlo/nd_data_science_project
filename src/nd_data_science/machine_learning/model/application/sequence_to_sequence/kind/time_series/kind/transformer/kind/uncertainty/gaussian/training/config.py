from typing import Dict, Any
from nd_math.probability.statistic.population.sampling.sampler.kind.countable.finite.members_mentioned.numbered.sequence.sliding_window.sliding_window import \
    SlidingWindow


class Config:
    def __init__(
            self,
            training_sequence_size: int,
            input_sequence_size: int,
            output_sequence_size: int,
            sequence_overlap_size: int,
            epochs: int,
            batch_size: int,
            learning_rate: float = 1e-3,
            shuffle: bool = True,
    ):
        self._training_sequence_size = training_sequence_size
        self._input_sequence_size = input_sequence_size
        self._output_sequence_size = output_sequence_size
        self._sequence_overlap_size = sequence_overlap_size
        self._epochs = int(epochs)
        self._batch_size = int(batch_size)
        self._learning_rate = float(learning_rate)
        self._shuffle = bool(shuffle)

        if self._epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self._batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self._learning_rate <= 0.0:
            raise ValueError("learning_rate must be > 0.")

    def get_epochs(self) -> int:
        return self._epochs

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def get_shuffle(self) -> bool:
        return self._shuffle

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": int(self._epochs),
            "batch_size": int(self._batch_size),
            "learning_rate": float(self._learning_rate),
            "shuffle": bool(self._shuffle),
        }

    def get_sliding_window(self) -> SlidingWindow:
        return SlidingWindow(self._input_sequence_size, self._output_sequence_size, self._sequence_overlap_size)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict.")

        return cls(
            training_sequence_size=int(payload["training_sequence_size"]),
            input_sequence_size=int(payload["input_sequence_size"]),
            output_sequence_size=int(payload["output_sequence_size"]),
            sequence_overlap_size=int(payload["sequence_overlap_size"]),
            epochs=int(payload["epochs"]),
            batch_size=int(payload["batch_size"]),
            learning_rate=float(payload["learning_rate"]),
            shuffle=bool(payload["shuffle"]),
        )

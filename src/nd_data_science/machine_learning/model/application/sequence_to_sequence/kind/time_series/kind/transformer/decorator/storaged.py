from nd_data_science.machine_learning.model.application.sequence_to_sequence.kind.time_series_forcating.kind.transformer.decorator.decorator import Decorator as TransformerTimeSeriesForcatingDecorator


class Storaged(TransformerTimeSeriesForcatingDecorator):
    def save_model(self, save_path: str) -> None:
        self.save(save_path)

    def load_transformer_model(self, save_path: str) -> "TransformerDraft":
        loaded_model = tf.keras.models.load_model(
            save_path,
            custom_objects={"TransformerDraft": TransformerDraft},
        )
        return loaded_model

    def save(self, save_path: str) -> None:
        self.save_model("transformer_sequence_to_sequence_saved_model.keras")

    def load(self) -> None:
        self.load_transformer_model("transformer_sequence_to_sequence_saved_model.keras")
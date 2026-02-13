import tensorflow as tf
from tensorflow.keras import layers as TfLayers, Model as TfModel


class Architecture:
    def __init__(
            self,
            model_dimension: int,
            number_of_attention_heads: int,
            feed_forward_dimension: int,
            input_feature_count: int,
            output_time_steps: int,
            output_feature_count: int,
            maximum_time_steps: int = 2048,
            dropout_rate: float = 0.1,
    ):
        self._model_dimension = int(model_dimension)
        self._number_of_attention_heads = int(number_of_attention_heads)
        self._feed_forward_dimension = int(feed_forward_dimension)
        self._input_feature_count = int(input_feature_count)
        self._output_time_steps = int(output_time_steps)
        self._output_feature_count = int(output_feature_count)
        self._maximum_time_steps = int(maximum_time_steps)
        self._dropout_rate = float(dropout_rate)

        if self._input_feature_count <= 0:
            raise ValueError("input_feature_count must be > 0.")
        if self._model_dimension <= 0:
            raise ValueError("model_dimension must be > 0.")
        if self._number_of_attention_heads <= 0:
            raise ValueError("number_of_attention_heads must be > 0.")
        if self._feed_forward_dimension <= 0:
            raise ValueError("feed_forward_dimension must be > 0.")
        if self._output_time_steps <= 0:
            raise ValueError("output_time_steps must be > 0.")
        if self._output_feature_count <= 0:
            raise ValueError("output_feature_count must be > 0.")
        if self._maximum_time_steps <= 0:
            raise ValueError("maximum_time_steps must be > 0.")
        if self._model_dimension % self._number_of_attention_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_attention_heads.")
        if self._dropout_rate < 0.0 or self._dropout_rate >= 1.0:
            raise ValueError("dropout_rate must be in [0.0, 1.0).")

    def get_model_dimension(self) -> int:
        return self._model_dimension

    def get_number_of_attention_heads(self) -> int:
        return self._number_of_attention_heads

    def get_feed_forward_dimension(self) -> int:
        return self._feed_forward_dimension

    def get_input_feature_count(self) -> int:
        return self._input_feature_count

    def get_output_time_steps(self) -> int:
        return self._output_time_steps

    def get_output_feature_count(self) -> int:
        return self._output_feature_count

    def get_maximum_time_steps(self) -> int:
        return self._maximum_time_steps

    def get_dropout_rate(self) -> float:
        return self._dropout_rate

    def build_tf_model(self) -> TfModel:
        """Build a minimal vanilla seq2seq Transformer (direct mapping, no decoder)."""
        d_model = self._model_dimension
        heads = self._number_of_attention_heads
        ff_dim = self._feed_forward_dimension
        f_in = self._input_feature_count
        f_out = self._output_feature_count
        t_out = self._output_time_steps
        max_steps = self._maximum_time_steps
        dropout_rate = self._dropout_rate

        per_head = d_model // heads

        x_in = TfLayers.Input(shape=(t_out, f_in), dtype=tf.float32, name="x_in")

        x = TfLayers.Dense(d_model, name="in_proj")(x_in)

        pos_emb = TfLayers.Embedding(max_steps, d_model, name="pos_emb")

        def add_positional(tensor):
            time_steps = tf.shape(tensor)[1]
            positions = pos_emb(tf.range(time_steps))
            positions = tf.expand_dims(positions, axis=0)
            return tensor + positions

        x = TfLayers.Lambda(add_positional, name="add_pos")(x)

        mha = TfLayers.MultiHeadAttention(
            num_heads=heads,
            key_dim=per_head,
            dropout=dropout_rate,
            name="mha",
        )
        attn = mha(query=x, value=x, key=x)
        attn = TfLayers.Dropout(dropout_rate, name="drop_after_attn")(attn)
        x = TfLayers.LayerNormalization(epsilon=1e-6, name="ln_after_attn")(x + attn)

        ff = tf.keras.Sequential(
            [TfLayers.Dense(ff_dim, activation="relu"), TfLayers.Dense(d_model)],
            name="ffn",
        )(x)
        ff = TfLayers.Dropout(dropout_rate, name="drop_after_ffn")(ff)
        x = TfLayers.LayerNormalization(epsilon=1e-6, name="ln_after_ffn")(x + ff)

        y = TfLayers.Dense(f_out, name="out_proj")(x)

        return TfModel(inputs=x_in, outputs=y, name="vanilla_seq2seq_transformer")

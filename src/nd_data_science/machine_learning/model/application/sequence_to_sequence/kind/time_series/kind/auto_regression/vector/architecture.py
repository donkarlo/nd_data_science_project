class Architecture:
    def __init__(
            self,
            feature_count: int,
            lag_order: int,
            include_intercept: bool = True,
    ):
        self._feature_count = int(feature_count)
        self._lag_order = int(lag_order)
        self._include_intercept = bool(include_intercept)

        if self._feature_count <= 0:
            raise ValueError("feature_count must be > 0.")
        if self._lag_order <= 0:
            raise ValueError("lag_order must be > 0.")

    def get_feature_count(self) -> int:
        return self._feature_count

    def get_lag_order(self) -> int:
        return self._lag_order

    def get_include_intercept(self) -> bool:
        return self._include_intercept
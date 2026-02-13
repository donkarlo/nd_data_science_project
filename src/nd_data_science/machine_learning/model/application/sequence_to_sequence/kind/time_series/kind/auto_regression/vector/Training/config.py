class Config:
    def __init__(
            self,
            fit_method: str = "ols",
            regularization_strength: float | None = None,
            select_lag_by: str | None = None,
    ):
        self._fit_method = str(fit_method)
        self._regularization_strength = regularization_strength
        self._select_lag_by = select_lag_by

        if self._fit_method not in {"ols", "ridge", "lasso"}:
            raise ValueError("fit_method must be one of {'ols', 'ridge', 'lasso'}.")

        if self._regularization_strength is not None and self._regularization_strength <= 0.0:
            raise ValueError("regularization_strength must be > 0.")

        if self._select_lag_by is not None and self._select_lag_by not in {"aic", "bic", "hqic"}:
            raise ValueError("select_lag_by must be one of {'aic', 'bic', 'hqic'}.")

    def get_select_lag_by(self)->str:
        return self._select_lag_by

    def get_fit_method(self)->str:
        return self._fit_method

    def get_regularization_strength(self) -> str:
        return self._regularization_strength

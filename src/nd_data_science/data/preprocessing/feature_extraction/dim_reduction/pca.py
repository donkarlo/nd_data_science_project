from data.preprocessing.feature_extraction.dim_reduction.dim_reduction_base import DimReductionBase


class Pca(DimReductionBase):
    def __init__(self, rows:np.ndarray):
        self.rows = rows
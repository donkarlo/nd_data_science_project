class DimReductionBase:
    def __init__(self, rows:np.ndarray):
        self._rows = rows

        #lazy loading
        self._resulted_rows = None

    @abstractmethod
    def get_resulted_rows(self)->np.ndarray:
        pass
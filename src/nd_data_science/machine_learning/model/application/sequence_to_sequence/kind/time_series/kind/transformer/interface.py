from abc import ABC, abstractmethod
from typing import Protocol
import numpy as np


class Interface(ABC, Protocol):
    @abstractmethod
    def get_forcat(self, time_serie_to_forcast:np.nd_array)->np.nd_array:
        pass
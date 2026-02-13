from nd_utility.data.kind.dic.dic import Dic


class Parameters:
    """
    Paramaters in a model are the variables in the model which should be learned_parameters
    """

    def __init__(self, parameter_values: Dic):
        self._parameter_values = parameter_values
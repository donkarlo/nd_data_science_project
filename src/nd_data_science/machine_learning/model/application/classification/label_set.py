from nd_data_science.machine_learning.model.application.classification.label import Label


class LabelSet:
    def __init__(self, labels:tuple[Label,...]):
        self._labels = labels
        self.__has_unique_labels()

    def __has_unique_labels(self):
        return False
from src.nd_data_science.machine_learning.classification.evaluation.eval_base import EvalBase

class BiClassEval(EvalBase):
    def __init__(self, tp:int, tn:int, fp:int, fn:int):
        self._tp = tp
        self._tn = tn
        self._fp = fp
        self._fn = fn

        #lazy loading
        self._accuracy = None
        self._precision = None
        self._recall = None
        self._f1_score = None
        self._confusion_matrix = None

    def get_accuracy(self):
        if self._accuracy is None:
            total = self._tp + self._tn + self._fp + self._fn
            self._accuracy = (self._tp + self._tn) / total if total != 0 else 0
        return self._accuracy

    def get_precision(self):
        if self._precision is None:
            denom = self._tp + self._fp
            self._precision = self._tp / denom if denom != 0 else 0
            return self._precision

    def get_recall(self):
        if self._recall is None:
            denom = self._tp + self._fn
            self._recall = self._tp / denom if denom != 0 else 0
        return self._recall

    def get_f1_score(self):
        if self._f1_score is None:
            precision = self.get_precision()
            recall = self.get_recall()
            denom = precision + recall
            self._f1_score = 2 * precision * recall / denom if denom != 0 else 0
        return self._f1_score

    def get_confusion_matrix(self):
        if self._confusion_matrix is None:
            self._confusion_matrix = [[self._tp, self._fn], [self._fp, self._tn]]
        return self._confusion_matrix
from allennlp.training.metrics import F1Measure

from allennlp.training.metrics.metric import Metric


@Metric.register("confusion_matrix")
class ConfusionMatrix(F1Measure):
    def get_metric(self, reset: bool = False):
        ret = list(map(int, (self._true_positives, self._true_negatives,
                             self._false_positives, self._false_negatives)))
        if reset:
            self.reset()
        return ret

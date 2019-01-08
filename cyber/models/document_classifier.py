from typing import Dict

import torch
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import F1Measure
from overrides import overrides

from cyber.metrics.confusion_matrix import ConfusionMatrix


class DocumentClassifier(Model):
    def __init__(self, *args, **kwargs) -> None:
        super(DocumentClassifier, self).__init__(*args, **kwargs)
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=1),
            "confusion_matrix": ConfusionMatrix(positive_label=1),
        }

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.metrics["f1"].get_metric(reset=reset)
        tp, tn, fp, fn = self.metrics["confusion_matrix"].get_metric(reset=reset)
        return {"accuracy": self.metrics["accuracy"].get_metric(reset=reset),
                "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

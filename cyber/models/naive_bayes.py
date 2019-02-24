from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from overrides import overrides
from sklearn.naive_bayes import BernoulliNB

from cyber.models.document_classifier import DocumentClassifier


@Model.register("naive_bayes")
class NaiveBayes(DocumentClassifier):
    def __init__(self, vocab: Vocabulary) -> None:
        super(NaiveBayes, self).__init__(vocab)

        self.vocab_size = self.vocab.get_vocab_size("tokens")
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.nb = BernoulliNB()
        # noinspection PyCallingNonCallable
        self.dummy = nn.Parameter(torch.tensor(0.0))

    @overrides
    def forward(self, text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        text_mask = util.get_text_field_mask(text).numpy()
        tokens = text["tokens"].numpy()
        bow = np.eye(self.vocab_size + 1, dtype=int)[text_mask * (tokens + 1)].sum(1)[:, 1:]
        self.nb.partial_fit(bow, label, classes=list(range(self.num_classes)))
        # noinspection PyCallingNonCallable
        log_proba = torch.tensor(self.nb.predict_log_proba(bow))

        output_dict = {"log_proba": log_proba}
        if label is not None:
            for metric in self.metrics.values():
                metric(log_proba, label)
            # noinspection PyCallingNonCallable
            output_dict["loss"] = torch.tensor(self.nb.score(bow, label)) + self.dummy

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        log_proba = output_dict["log_proba"]
        argmax_indices = np.argmax(log_proba, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.metrics["f1"].get_metric(reset=reset)
        tp, tn, fp, fn = self.metrics["confusion_matrix"].get_metric(reset=reset)
        return {"accuracy": self.metrics["accuracy"].get_metric(reset=reset),
                "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def state_dict(self, *args, **kwargs):
        return self.nb.get_params()

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        self.nb.set_params(**state_dict)

from typing import Dict, Optional, List, Any

import numpy as np
import torch
import torch.nn as nn
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
from overrides import overrides
from sklearn.svm import SVC

from cyber.models.document_classifier import DocumentClassifier


@Model.register("svm")
class Svm(DocumentClassifier):

    def __init__(self, vocab: Vocabulary) -> None:
        super(Svm, self).__init__(vocab)

        self.vocab_size = self.vocab.get_vocab_size("tokens")
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.svm = SVC(gamma='scale', cache_size=4000, max_iter=-1, tol=1e-5)
        # self.svm = SVC(kernel="poly", gamma='scale', cache_size=4000, max_iter=-1)
        # self.svm = SVC(kernel="linear", gamma='scale', cache_size=4000, max_iter=-1)
        self.dummy = nn.Parameter(torch.tensor(0.0))

    @overrides
    def forward(self, text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: Optional[List[Dict[str, Any]]]=None) -> Dict[str, torch.Tensor]:
        text_mask = util.get_text_field_mask(text)
        tokens = text["tokens"]
        bow = torch.zeros((tokens.size(0), self.vocab_size))
        bow.scatter_(1, tokens, torch.ones((tokens.size(0), self.vocab_size)))
        mask = torch.zeros((tokens.size(0), self.vocab_size))
        mask.scatter_(1, text_mask, torch.ones(
            (text_mask.size(0), self.vocab_size)))
        bow *= mask
        bow = bow.numpy()
        # label = torch.ones(label.size())
        # label[-1] = 0
        self.svm.fit(bow, label)  # , classes=list(range(self.num_classes)))
        predict = torch.tensor(self.svm.predict(bow))
        # print("----")
        # print("labels", label)
        # print(predict)
        # labels_num = predict.max()+1
        labels_num = 2
        predict = torch.zeros(len(predict), labels_num).scatter_(1, predict.unsqueeze(1), 1.)
        # print(predict)

        # print("shapes", predict.shape, bow.shape, label.shape)
        # raise
        output_dict = {"predict": predict}
        if label is not None:
            for metric in self.metrics.values():
                metric(predict, label)
            output_dict["loss"] = torch.tensor(
                self.svm.score(bow, label)) + self.dummy

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predict = output_dict["predict"]
        # argmax_indices = np.argmax(predict, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in predict]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.metrics["f1"].get_metric(reset=reset)
        tp, tn, fp, fn = self.metrics[
            "confusion_matrix"].get_metric(reset=reset)
        return {"accuracy": self.metrics["accuracy"].get_metric(reset=reset),
                "precision": precision, "recall": recall, "f1": f1,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn}

    def state_dict(self, *args, **kwargs):
        return self.svm.get_params()

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        self.svm.set_params(**state_dict)

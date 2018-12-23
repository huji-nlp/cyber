from typing import Dict, Optional, Union

import numpy
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Maxout, TextFieldEmbedder
from allennlp.modules import Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import F1Measure
from overrides import overrides

from cyber.metrics.confusion_matrix import ConfusionMatrix


@Model.register("document_classifier")
class DocumentClassifier(Model):
    """
    This ``Model`` performs text classification.  We assume we're given a
    text and we predict some output label.
    The basic model structure: we'll embed the text and encode it with
    a Seq2VecEncoder, getting a single vector representing the content.  We'll then feed
    the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model_text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    internal_text_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the input text to a vector.
    output_layer : ``Union[Maxout, FeedForward]``
        The maxout or feed forward network that takes the final representations and produces
        a classification prediction.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 model_text_field_embedder: TextFieldEmbedder,
                 internal_text_encoder: Seq2VecEncoder,
                 output_layer: Union[FeedForward, Maxout],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(DocumentClassifier, self).__init__(vocab, regularizer)

        self.model_text_field_embedder = model_text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.internal_text_encoder = internal_text_encoder
        self.output_layer = output_layer

        if model_text_field_embedder.get_output_dim() != internal_text_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the model_text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(model_text_field_embedder.get_output_dim(),
                                                            internal_text_encoder.get_input_dim()))
        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(positive_label=1),
            "confusion_matrix": ConfusionMatrix(positive_label=1),
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.model_text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.internal_text_encoder(embedded_text, text_mask)

        logits = self.output_layer(encoded_text)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'])
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
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

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'DocumentClassifier':
        embedder_params = params.pop("model_text_field_embedder")
        model_text_field_embedder = TextFieldEmbedder.from_params(embedder_params, vocab=vocab)
        internal_text_encoder = Seq2VecEncoder.from_params(params.pop("internal_text_encoder"))
        output_layer = FeedForward.from_params(params.pop("output_layer"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   model_text_field_embedder=model_text_field_embedder,
                   internal_text_encoder=internal_text_encoder,
                   output_layer=output_layer,
                   initializer=initializer,
                   regularizer=regularizer)

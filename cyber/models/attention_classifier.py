from typing import Dict, Optional, List, Any, Union

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Elmo, FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from overrides import overrides
from torch import nn

from cyber.metrics.attention import AttentionMetric
from cyber.models.document_classifier import DocumentClassifier


# noinspection PyProtectedMember
@Model.register("attention_classifier")
class AttentionClassifier(DocumentClassifier):
    """
    This class implements the Biattentive Classification Network model described
    in section 5 of `Learned in Translation: Contextualized Word Vectors (NIPS 2017)
    <https://arxiv.org/abs/1708.00107>`_ for text classification. We assume we're
    given a piece of text, and we predict some output label.

    At a high level, the model starts by embedding the tokens and running them through
    a feed-forward neural net (``pre_encode_feedforward``). Then, we encode these
    representations with a ``Seq2SeqEncoder`` (``encoder``). We run biattention
    on the encoder output representations (self-attention in this case, since
    the two representations that typically go into biattention are identical) and
    get out an attentive vector representation of the text. We combine this text
    representation with the encoder outputs computed earlier, and then run this through
    yet another ``Seq2SeqEncoder`` (the ``integrator``). Lastly, we take the output of the
    integrator and max, min, mean, and self-attention pool to create a final representation,
    which is passed through some maxout layers
    to output a classification (``output_layer``).

    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    embedding_dropout : ``float``
        The amount of dropout to apply on the embeddings.
    pre_encode_feedforward : ``FeedForward``
        A feedforward network that is run on the embedded tokens before they
        are passed to the encoder.
    encoder : ``Seq2SeqEncoder``
        The encoder to use on the tokens.
    integrator : ``Seq2SeqEncoder``
        The encoder to use when integrating the attentive text encoding
        with the token encodings.
    integrator_dropout : ``float``
        The amount of dropout to apply on integrator output.
    output_layer : ``Maxout``
        The maxout network that takes the final representations and produces
        a classification prediction.
    elmo : ``Elmo``, optional (default=``None``)
        If provided, will be used to concatenate pretrained ELMo representations to
        either the integrator output (``use_integrator_output_elmo``) or the
        input (``use_input_elmo``).
    use_input_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the input vectors.
    use_integrator_output_elmo : ``bool`` (default=``False``)
        If true, concatenate pretrained ELMo representations to the integrator output.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 embedding_dropout: float,
                 pre_encode_feedforward: FeedForward,
                 encoder: Seq2SeqEncoder,
                 integrator: Seq2SeqEncoder,
                 integrator_dropout: float,
                 output_layer: Maxout,
                 elmo: Elmo,
                 use_input_elmo: bool = False,
                 use_integrator_output_elmo: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AttentionClassifier, self).__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        if "elmo" in self._text_field_embedder._token_embedders:
            raise ConfigurationError("To use ELMo in the AttentionClassifier input, "
                                     "remove elmo from the text_field_embedder and pass an "
                                     "Elmo object to the AttentionClassifier and set the "
                                     "'use_input_elmo' and 'use_integrator_output_elmo' flags accordingly.")
        self._embedding_dropout = nn.Dropout(embedding_dropout)
        self._num_classes = self.vocab.get_vocab_size("labels")

        self._pre_encode_feedforward = pre_encode_feedforward
        self._encoder = encoder
        self._integrator = integrator
        self._integrator_dropout = nn.Dropout(integrator_dropout)

        self._elmo = elmo
        self._use_input_elmo = use_input_elmo
        self._use_integrator_output_elmo = use_integrator_output_elmo
        self._num_elmo_layers = int(self._use_input_elmo) + int(self._use_integrator_output_elmo)

        # Calculate combined integrator output dim, taking into account elmo
        self._combined_integrator_output_dim = self._integrator.get_output_dim()
        if self._use_integrator_output_elmo:
            self._combined_integrator_output_dim += self._elmo.get_output_dim()

        self._self_attentive_pooling_projection = nn.Linear(self._combined_integrator_output_dim, 1)
        self._output_layer = output_layer

        self.check_input()

        self.loss = torch.nn.CrossEntropyLoss()
        initializer(self)

        self._attention_metric = AttentionMetric()

    def check_input(self):
        if self._elmo is None:  # Check that, if elmo is None, none of the elmo flags are set.
            if self._num_elmo_layers:
                raise ConfigurationError("One of 'use_input_elmo' or 'use_integrator_output_elmo' is True, "
                                         "but no Elmo object was provided upon construction. Pass in an Elmo "
                                         "object to use Elmo.")
        else:  # Check that, if elmo is not None, we use it somewhere.
            if not self._num_elmo_layers:
                raise ConfigurationError("Elmo object provided upon construction, but both 'use_input_elmo' "
                                         "and 'use_integrator_output_elmo' are 'False'. Set one of them to "
                                         "'True' to use Elmo, or do not provide an Elmo object upon construction.")
            # Check that the number of flags set is equal to the num_output_representations of the Elmo object
            if len(self._elmo._scalar_mixes) != self._num_elmo_layers:
                raise ConfigurationError("Elmo object has num_output_representations=%s, but this does not "
                                         "match the number of use_*_elmo flags set to true. use_input_elmo "
                                         "is %s, and use_integrator_output_elmo is %s".format(
                                            len(self._elmo._scalar_mixes), self._use_input_elmo,
                                            self._use_integrator_output_elmo))
        check_dimensions_match(self._text_field_embedder.get_output_dim() +
                               (self._elmo.get_output_dim() if self._use_input_elmo else 0),
                               self._pre_encode_feedforward.get_input_dim(),
                               "text field embedder output dim + ELMo output dim", "Pre-encoder feedforward input dim")
        check_dimensions_match(self._pre_encode_feedforward.get_output_dim(), self._encoder.get_input_dim(),
                               "Pre-encoder feedforward output dim", "Encoder input dim")
        check_dimensions_match(self._encoder.get_output_dim() * 3,
                               self._integrator.get_input_dim(),
                               "Encoder output dim * 3",
                               "Integrator input dim")
        check_dimensions_match(self._combined_integrator_output_dim * 4, self._output_layer.get_input_dim(),
                               "(Integrator output dim + ELMo output dim) * 4", "Output layer input dim")
        check_dimensions_match(self._output_layer.get_output_dim(), self._num_classes,
                               "Output layer output dim", "Number of classes.")

    @overrides
    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None,
                metadata: Optional[List[Dict[str, Any]]] = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : Dict[str, torch.LongTensor], required
            The output of ``TextField.as_array()``.
        label : torch.LongTensor, optional (default = None)
            A variable representing the label for each instance in the batch.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the document tokens.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a
            distribution over the label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        self_weights : torch.FloatTensor
            Attention weights.
        tokens : List, optional
            Tokens for each instance.

        """
        text_mask = util.get_text_field_mask(text).float()
        elmo_tokens = text.pop("elmo", None)  # Pop elmo tokens, since elmo embedder should not be present.
        embedded_text = self._text_field_embedder(text) if text else None
        batch_size = embedded_text.size(0)

        if elmo_tokens is not None:  # Add the "elmo" key back to "tokens" if not None, since the tests and the
            text["elmo"] = elmo_tokens  # subsequent training epochs rely not being modified during forward()

        input_elmo = integrator_output_elmo = None  # Create ELMo embeddings if applicable
        if self._elmo:
            if elmo_tokens is None:
                raise ConfigurationError("Model was built to use Elmo, but input text is not tokenized for Elmo.")
            elmo_representations = self._elmo(elmo_tokens)["elmo_representations"]
            if self._use_integrator_output_elmo:
                integrator_output_elmo = elmo_representations.pop()  # Pop from the end is more performant with list
            if self._use_input_elmo:
                input_elmo = elmo_representations.pop()
            assert not elmo_representations

        if self._use_input_elmo:
            embedded_text = input_elmo if embedded_text is None else torch.cat([embedded_text, input_elmo], dim=-1)

        dropped_embedded_text = self._embedding_dropout(embedded_text)
        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)
        encoded_tokens = self._encoder(pre_encoded_text, text_mask)

        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())  # Compute biattention
        attention_weights = util.masked_softmax(attention_logits, text_mask)  # This is a special case,
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)  # since the inputs are the same

        integrator_input = torch.cat([encoded_tokens,  # Build the input to the integrator
                                      encoded_tokens - encoded_text,
                                      encoded_tokens * encoded_text], 2)
        integrated_encodings = self._integrator(integrator_input, text_mask)

        # Concatenate ELMo representations to integrated_encodings if specified
        if self._use_integrator_output_elmo:
            integrated_encodings = torch.cat([integrated_encodings, integrator_output_elmo], dim=-1)

        max_masked_integrated_encodings = util.replace_masked_values(  # Simple Pooling layers
            integrated_encodings, text_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length)
        self_attentive_logits = self._self_attentive_pooling_projection(integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)

        logits = self._output_layer(pooled_representations_dropped)
        class_probabilities = F.softmax(logits, dim=-1)

        output_dict = {
            "logits": logits,
            "class_probabilities": class_probabilities,
            "attention_weights": attention_weights,
            "self_weights": self_weights,
        }
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        if metadata is not None:
            tokens = [metadata[i]["tokens"] for i in range(batch_size)]
            output_dict["tokens"] = tokens
            labels = [self.vocab.get_token_from_index(x, namespace="labels") for x in
                      numpy.argmax(class_probabilities.cpu().data.numpy(), axis=-1)]
            self._attention_metric(tokens, self_weights, labels, text_mask)

        return output_dict

    # noinspection PyTypeChecker
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, List[Any]]]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict["class_probabilities"].cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        batch_size = len(output_dict["tokens"])
        output_dict["label"] = [self.vocab.get_token_from_index(x, namespace="labels") for x in argmax_indices]
        output_dict["all_labels"] = batch_size * [
            [v for k, v in sorted(self.vocab.get_index_to_token_vocabulary("labels").items())]
        ]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset=reset)
        metrics.update(self._attention_metric.get_metric(reset=reset))
        return metrics

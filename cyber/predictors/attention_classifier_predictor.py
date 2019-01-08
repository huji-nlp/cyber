from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from overrides import overrides


@Predictor.register('attention_classifier')
class AttentionClassifierPredictor(Predictor):
    """Predictor wrapper for the AttentionClassifier"""

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(text=json_dict['text_input'])

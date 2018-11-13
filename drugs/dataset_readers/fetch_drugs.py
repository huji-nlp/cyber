import logging
import os
from typing import Dict

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CATEGORIES = ["ebay", "onion"]
TRAIN_RATIO = .9
MAX_LENGTH = 999999


@DatasetReader.register("drugs")
class DrugsDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        logger.info("Reading %s instance(s)", file_path)
        drugs_data = self.fetch_drugs(subset=file_path, categories=CATEGORIES) if file_path in ("train", "test") else \
            [(self.read_file(file_path), None)]
        for text, target in drugs_data:
            yield self.text_to_instance(text, target)

    @overrides
    def text_to_instance(self, text: str, target: int = None) -> Instance:
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if target is not None:
            fields['label'] = LabelField(int(target), skip_indexing=True)
        return Instance(fields)

    @staticmethod
    def fetch_drugs(subset, categories):
        for i, category in enumerate(categories):
            files = sorted(os.listdir(category))
            num_train = int(TRAIN_RATIO * len(files))
            files = files[:num_train] if subset == "train" else files[num_train:]
            for filename in files:
                path = os.path.join(category, filename)
                yield DrugsDatasetReader.read_file(path), i

    @staticmethod
    def read_file(path):
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")[:MAX_LENGTH]

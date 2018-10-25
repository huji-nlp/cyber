import logging
from itertools import islice
from typing import Dict

from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides
from sklearn.datasets import fetch_20newsgroups

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CATEGORIES = ["comp.graphics", "sci.space", "rec.sport.baseball"]


@DatasetReader.register("20newsgroups")
class NewsgroupsDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        if file_path not in ("train", "test"):
            raise ConfigurationError("Path string not specified in read method")

        logger.info("Reading instances from: %s", file_path)
        newsgroups_data = fetch_20newsgroups(subset=file_path, categories=CATEGORIES)

        for text, target in islice(zip(newsgroups_data.data, newsgroups_data.target),
                                   400 if file_path == "validate" else None):
            yield self.text_to_instance(text, target)

    @overrides
    def text_to_instance(self, text: str, target: str = None) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_text = self._tokenizer.tokenize(text)
        text_field = TextField(tokenized_text, self._token_indexers)
        fields = {'text': text_field}
        if target is not None:
            fields['label'] = LabelField(int(target), skip_indexing=True)
        return Instance(fields)

import logging
import os
from typing import Dict, Tuple

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CATEGORIES = ("ebay", "onion/legal", "onion/illegal")


@DatasetReader.register("drugs")
class DrugsDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None,
                 categories: Tuple[str] = CATEGORIES, train_ratio: float = .9, max_length: int = 999999) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._categories = categories
        self._train_ratio = train_ratio
        self._max_length = max_length

    @overrides
    def _read(self, file_path):
        logger.info("Reading %s instance(s)", file_path)
        drugs_data = self.fetch_drugs(subset=file_path, categories=self._categories) if file_path in ("train", "test") \
            else [(self.read_file(file_path), None)]
        for text, target in drugs_data:
            yield self.text_to_instance(text, target)

    @overrides
    def text_to_instance(self, text: str, target: int = None) -> Instance:
        text_field = TextField(self._tokenizer.tokenize(text), self._token_indexers)
        fields: Dict[str, Field] = {"text": text_field}
        if target is not None:
            fields["label"] = LabelField(target)
        return Instance(fields)

    def fetch_drugs(self, subset, categories):
        files = [sorted(os.listdir(category)) for category in categories]
        num_files_per_category = min(map(len, files))  # Take the same number of files from each category
        num_train = int(self._train_ratio * num_files_per_category)
        for category, category_files in zip(categories, files):
            for filename in category_files[:num_train] if subset == "train" else \
                    category_files[num_train:num_files_per_category]:
                yield self.read_file(os.path.join(category, filename)), category

    def read_file(self, path):
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")[:self._max_length]

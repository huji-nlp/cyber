import logging
import os
from typing import Dict, Tuple

from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CATEGORIES = ("ebay", "onion/legal", "onion/illegal")


@DatasetReader.register("document")
class DocumentDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None,
                 categories: Tuple[str] = CATEGORIES, train_ratio: float = .8, validation_ratio: float = .1,
                 max_length: int = 999, mask: str = None) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._categories = categories
        self._train_ratio = train_ratio
        self._validation_ratio = validation_ratio
        self._max_length = max_length
        self._mask = mask

    @overrides
    def _read(self, file_path):
        logger.info("Reading %s instance(s)", file_path)
        drugs_data = self.fetch_documents(subset=file_path, categories=self._categories)\
            if file_path in ("train", "validation", "test") \
            else [(self.read_file(file_path), None)]
        for text, target in drugs_data:
            yield self.text_to_instance(text, target)

    @overrides
    def text_to_instance(self, text: str, target: int = None) -> Instance:
        text_field = TextField(self._tokenizer.tokenize(text), self._token_indexers)
        text_field.tokens = [self.mask_token(token) for token in text_field.tokens]
        fields: Dict[str, Field] = {"text": text_field}
        if target is not None:
            fields["label"] = LabelField(target)
        return Instance(fields)

    def mask_token(self, token):
        if self._mask:
            if token.pos_ in self._mask:
                return Token(text=token.pos_, idx=token.idx, lemma=token.pos_, pos=token.pos_, tag=token.tag_,
                             dep=token.dep_, ent_type=token.ent_type_)
        return token

    def fetch_documents(self, subset, categories):
        # Take the same number of lines from each category
        lines_per_category = list(zip(*map(self.fetch_lines, categories)))
        num_train = int(self._train_ratio * len(lines_per_category))
        num_validation = int(self._validation_ratio * len(lines_per_category))
        print("Categories: " + ", ".join(categories))
        print("Using train/validation/test split of %d/%d/%d lines for each category" % (
            num_train, num_validation, len(lines_per_category) - num_train - num_validation))
        for category_lines in zip(*lines_per_category):
            if subset == "train":
                start, end = 0, num_train
            elif subset == "validation":
                start, end = num_train, num_train + num_validation
            elif subset == "test":
                start, end = num_train + num_validation, len(category_lines)
            else:
                raise ValueError("Invalid subset: %s" % subset)
            yield from category_lines[start:end]

    def fetch_lines(self, category):
        for filename in sorted(os.listdir(os.path.join(*[c + "_clean" for c in os.path.split(category)]))):
            for line in self.read_file(os.path.join(category, filename)):
                yield line, category

    def read_file(self, path):
        with open(path, "rb") as f:
            for line in f.read().decode("utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line:
                    yield line[:self._max_length]

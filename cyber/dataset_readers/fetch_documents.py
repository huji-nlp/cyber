import logging
import os
from typing import Dict, Tuple, Optional

from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

from cyber.util.clean_text import read_file, clean_lines
from cyber.util.split_data import DATA_SUBDIRS, clean_file_path

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("document")
class DocumentDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 categories: Optional[Tuple[str]] = None, mask: Optional[Tuple[str]] = None,
                 drop_masked: bool = False) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._categories = [c.split(os.sep) for c in categories] or DATA_SUBDIRS
        self._mask = mask
        self._drop_masked = drop_masked

    @overrides
    def _read(self, file_path):
        logger.info("Reading %s instance(s)", file_path)
        drugs_data = self.fetch_documents(subset=file_path, categories=self._categories)\
            if file_path in ("train", "validation", "test") else [(l, None) for l in clean_lines(read_file(file_path))]
        for text, target in drugs_data:
            yield self.text_to_instance(text, target)

    @overrides
    def text_to_instance(self, text: str, target: int = None) -> Instance:
        text_field = TextField(self._tokenizer.tokenize(text), self._token_indexers)
        metadata = {
            "tokens": [token.text for token in text_field.tokens],
        }
        text_field.tokens = list(filter(None, (self.mask_token(token) for token in text_field.tokens)))
        metadata["masked_tokens"] = [token.text for token in text_field.tokens]
        fields: Dict[str, Field] = {"text": text_field}
        if target is not None:
            fields["label"] = LabelField(target)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def fetch_documents(subset, categories):
        for category in categories:
            with open(clean_file_path(subdir=category, div=subset), encoding="utf-8") as f:
                for line in f:
                    yield line, "/".join(category)

    def mask_token(self, token):
        if self._mask is None:
            return token
        if not self._mask or token.pos_ in self._mask:
            if self._drop_masked:
                return None
            return Token(text=token.pos_, idx=token.idx, lemma=token.pos_, pos=token.pos_, tag=token.tag_,
                         dep=token.dep_, ent_type=token.ent_type_)

import logging
from typing import Dict, Tuple, Optional

from allennlp.data import Token
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

CATEGORIES = ("ebay", "illegal", "legal")


@DatasetReader.register("document")
class DocumentDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 mask: Optional[Tuple[str]] = None, drop_masked: bool = False) -> None:
        super().__init__()
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._mask = mask
        self._drop_masked = drop_masked

    @overrides
    def _read(self, file_paths):
        logger.info("Reading instances from " + ", ".join(file_paths))
        for line, category in self.fetch_documents(file_paths):
            yield self.text_to_instance(line, category)

    @overrides
    def text_to_instance(self, text: str, target: int = None) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        masked_tokens = list(filter(None, (self.mask_token(token) for token in tokens)))
        text_field = TextField(masked_tokens, self._token_indexers)
        metadata = {"tokens": [token.text for token in tokens],
                    "masked_tokens": [token.text for token in masked_tokens]}
        fields: Dict[str, Field] = {"text": text_field}
        if target is not None:
            fields["label"] = LabelField(target)
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)

    @staticmethod
    def fetch_documents(file_paths):
        for file_path in file_paths:
            category = next((c for c in CATEGORIES if c in file_path), None)
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    yield line, category

    def mask_token(self, token):
        if self._mask is None or token.pos_ not in self._mask:
            return token
        if self._drop_masked:
            return None
        return Token(text=token.pos_, idx=token.idx, lemma=token.pos_, pos=token.pos_, tag=token.tag_, dep=token.dep_,
                     ent_type=token.ent_type_)

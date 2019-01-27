from collections import Counter, defaultdict
from itertools import repeat
from typing import Optional, List

import torch
from allennlp.training.metrics.metric import Metric
from overrides import overrides


@Metric.register("attention")
class AttentionMetric(Metric):
    def __init__(self) -> None:
        self._cumulative_attention = defaultdict(float)
        self._occurrences = Counter()

    @overrides
    def __call__(self,
                 tokens: List[List[str]],
                 self_weights: torch.Tensor,
                 labels: List[torch.Tensor],
                 mask: Optional[torch.Tensor] = None):
        self_weights, mask = self.unwrap_to_tensors(self_weights, mask)

        for instance_tokens, instance_weights, label, instance_mask in zip(tokens, self_weights, labels,
                                                                           repeat(None) if mask is None else mask):
            for token, weight, token_mask in zip(instance_tokens, instance_weights, instance_mask):
                if mask is None or token_mask:
                    key = "\t".join((label, token))
                    self._cumulative_attention[key] += weight.item()
                    self._occurrences[key] += 1

    @overrides
    def get_metric(self, reset: bool = False):
        if self._occurrences:
            with open("attention.tsv", "w", encoding="utf-8") as f:
                for word, occurrences in self._occurrences.most_common():
                    print(word, occurrences, self._cumulative_attention[word], sep="\t", file=f)
        average_attention = Counter(**{k: v / self._occurrences[k] for k, v in self._cumulative_attention.items()})
        if reset:
            self.reset()
        return average_attention.most_common(1)

    @overrides
    def reset(self):
        self._cumulative_attention.clear()
        self._occurrences.clear()

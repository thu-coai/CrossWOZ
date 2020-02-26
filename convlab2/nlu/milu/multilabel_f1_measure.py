# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from typing import Dict, List, Optional, Set

import torch
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric


@Metric.register("multilabel_f1")
class MultiLabelF1Measure(Metric):
    """
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 namespace: str = "intent_labels",
                 ignore_classes: List[str] = None,
                 coarse: bool = True) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the label namespace.
        namespace : str, required.
            The vocabulary namespace for labels.
        ignore_classes : List[str], optional.
            Labels which will be ignored when computing metrics.
        """
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(namespace)
        self._ignore_classes: List[str] = ignore_classes or []
        self._coarse = coarse

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        if self._coarse:
            num_positives = predictions.sum()
            num_false_positives = ((predictions - gold_labels) > 0).long().sum()
            self._false_positives["coarse_overall"] += num_false_positives 
            num_true_positives = num_positives - num_false_positives
            self._true_positives["coarse_overall"] += num_true_positives 
            num_false_negatives = ((gold_labels - predictions) > 0).long().sum()
            self._false_negatives["coarse_overall"] += num_false_negatives 
        else:
            # Iterate over timesteps in batch.
            batch_size = gold_labels.size(0)
            for i in range(batch_size):
                prediction = predictions[i, :]
                gold_label = gold_labels[i, :]
                for label_id in range(gold_label.size(-1)):
                    label = self._label_vocabulary[label_id]
                    if prediction[label_id] == 1 and gold_label[label_id] == 1:
                        self._true_positives[label] += 1
                    elif prediction[label_id] == 1 and gold_label[label_id] == 0:
                        self._false_positives[label] += 1
                    elif prediction[label_id] == 0 and gold_label[label_id] == 1:
                        self._false_negatives[label] += 1


    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_labels: Set[str] = set()
        all_labels.update(self._true_positives.keys())
        all_labels.update(self._false_positives.keys())
        all_labels.update(self._false_negatives.keys())
        all_metrics = {}
        for label in all_labels:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[label],
                                                                  self._false_positives[label],
                                                                  self._false_negatives[label])
            precision_key = "precision" + "-" + label 
            recall_key = "recall" + "-" + label 
            f1_key = "f1-measure" + "-" + label 
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure

        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = float(true_positives) / float(true_positives + false_negatives)if true_positives + false_negatives > 0 else 0
        f1_measure = 2. * ((precision * recall) / (precision + recall)) if precision + recall > 0 else 0
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

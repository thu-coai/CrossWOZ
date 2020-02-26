# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, List, Any

from allennlp.training.metrics.metric import Metric


class DialogActItemF1Measure(Metric):
    """
    """
    def __init__(self) -> None:
        """
        Parameters
        ----------
        """
        # These will hold per label span counts.
        self._true_positives = 0 
        self._false_positives = 0 
        self._false_negatives = 0 


    def __call__(self,
                 predictions: List[Dict[str, Any]],
                 gold_labels: List[Dict[str, Any]]):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        """
        for prediction, gold_label in zip(predictions, gold_labels): 
            for dat in prediction:
                for sv in prediction[dat]:
                    if dat not in gold_label or sv not in gold_label[dat]:
                        self._false_positives += 1
                    else:
                        self._true_positives += 1
            for dat in gold_label:
                for sv in gold_label[dat]:
                    if dat not in prediction or sv not in prediction[dat]:
                        self._false_negatives += 1


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
        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(self._true_positives,
                                                              self._false_positives,
                                                              self._false_negatives)
        metrics = {}
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1-measure"] = f1_measure
        if reset:
            self.reset()
        return metrics


    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure


    def reset(self):
        self._true_positives = 0 
        self._false_positives = 0 
        self._false_negatives = 0 

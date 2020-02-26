import time
from collections import OrderedDict


class NumericMetric(object):
    """Base class for a numeric metric."""
    def __init__(self):
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def record(self, k, n=1):
        self.k += k
        self.n += n

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.k / self.n


class AverageMetric(NumericMetric):
    """Average."""
    def show(self):
        return '%.2f' % (1. * self.value())


class PercentageMetric(NumericMetric):
    """Percentage."""
    def show(self):
        return '%2.2f%%' % (100. * self.value())


class TimeMetric(object):
    """Time based metric."""
    def __init__(self):
        self.t = 0
        self.n = 0

    def reset(self):
        self.last_t = time.time()

    def record(self, n=1):
        self.t += time.time() - self.last_t
        self.n += 1

    def value(self):
        self.n = max(1, self.n)
        return 1.0 * self.t / self.n

    def show(self):
        return '%.3fs' % (1. * self.value())


class UniquenessMetric(object):
    """Metric that evaluates the number of unique sentences."""
    def __init__(self):
        self.seen = set()

    def reset(self):
        pass

    def record(self, sen):
        self.seen.add(' '.join(sen))

    def value(self):
        return len(self.seen)

    def show(self):
        return str(self.value())


class TextMetric(object):
    """Text based metric."""
    def __init__(self, text):
        self.text = text
        self.k = 0
        self.n = 0

    def reset(self):
        pass

    def value(self):
        self.n = max(1, self.n)
        return 1. * self.k / self.n

    def show(self):
        return '%.2f' % (1. * self.value())


class NGramMetric(TextMetric):
    """Metric that evaluates n grams."""
    def __init__(self, text, ngram=-1):
        super(NGramMetric, self).__init__(text)
        self.ngram = ngram

    def record(self, sen):
        n = len(sen) if self.ngram == -1 else self.ngram
        for i in range(len(sen) - n + 1):
            self.n += 1
            target = ' '.join(sen[i:i + n])
            if self.text.find(target) != -1:
                self.k += 1


class MetricsContainer(object):
    """A container that stores and updates several metrics."""
    def __init__(self):
        self.metrics = OrderedDict()

    def _register(self, name, ty, *args, **kwargs):
        name = name.lower()
        assert name not in self.metrics
        self.metrics[name] = ty(*args, **kwargs)

    def register_average(self, name, *args, **kwargs):
        self._register(name, AverageMetric, *args, **kwargs)

    def register_time(self, name, *args, **kwargs):
        self._register(name, TimeMetric, *args, **kwargs)

    def register_percentage(self, name, *args, **kwargs):
        self._register(name, PercentageMetric, *args, **kwargs)

    def register_ngram(self, name, *args, **kwargs):
        self._register(name, NGramMetric, *args, **kwargs)

    def register_uniqueness(self, name, *args, **kwargs):
        self._register(name, UniquenessMetric, *args, **kwargs)

    def record(self, name, *args, **kwargs):
        name = name.lower()
        assert name in self.metrics
        self.metrics[name].record(*args, **kwargs)

    def reset(self):
        for m in self.metrics.values():
            m.reset()

    def value(self, name):
        return self.metrics[name].value()

    def show(self):
        return ' '.join(['%s=%s' % (k, v.show()) for k, v in self.metrics.iteritems()])

    def dict(self):
        d = OrderedDict()
        for k, v in self.metrics.items():
            d[k] = v.show()
        return d

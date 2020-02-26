import numpy as np


class BaseDataLoaders(object):
    def __init__(self, name):
        self.data_size = None
        self.indexes = None
        self.name = name

    def _shuffle_indexes(self):
        np.random.shuffle(self.indexes)

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def epoch_init(self, config, shuffle=True, verbose=True, fix_batch=False):
        self.ptr = 0
        self.batch_size = config.batch_size
        self.num_batch = self.data_size // config.batch_size

        if verbose:
            print('Number of left over sample = %d' % (self.data_size - config.batch_size * self.num_batch))

        if shuffle and not fix_batch:
            self._shuffle_indexes()

        self.batch_indexes = []
        for i in range(self.num_batch):
            self.batch_indexes.append(self.indexes[i*self.batch_size: (i+1)*self.batch_size])

        if shuffle and fix_batch:
            self._shuffle_batch_indexes()

        if verbose:
            print('%s begins with %d batches' % (self.name, self.num_batch))

    def next_batch(self):
        if self.ptr < self.num_batch:
            selected_ids = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(selected_index=selected_ids)
        else:
            return None

    def _prepare_batch(self, *args, **kwargs):
        raise NotImplementedError('Have to override _prepare_batch()')

    def pad_to(self, max_len, tokens, do_pad):
        if len(tokens) >= max_len:
            return tokens[: max_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (max_len - len(tokens))
        else:
            return tokens

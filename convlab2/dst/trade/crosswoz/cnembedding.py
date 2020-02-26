import os
from os.path import join, dirname, abspath
import random

root_path = dirname(abspath(__file__))

class CNEmbedding:
    def __init__(self):
        vector_path = join(root_path, 'data', 'crosswoz', 'vector.txt')
        self.word2vec = {}
        with open(vector_path) as fin:
            lines = fin.readlines()[1:]
            for line in lines:
                line = line.strip()
                tokens = line.split(' ')
                word = tokens[0]
                vec = tokens[1:]
                vec = [float(item) for item in vec]
                self.word2vec[word] = vec
        self.embed_size = 100


    def emb(self, token, default='zero'):
        get_default = {
            'none': lambda: None,
            'zero': lambda: 0.,
            'random': lambda: random.uniform(-0.1, 0.1),
        }[default]
        vec = self.word2vec.get(token, None)
        if vec is None:
            vec = [get_default()] * self.embed_size
        return vec
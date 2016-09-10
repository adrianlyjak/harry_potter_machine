import random
from word2vec_api import W2V
from nltk import tokenize
from itertools import islice

class Text(object):
    def __init__(self, fname, model: W2V):
        self.fname = fname
        self.random = random.Random()
        with open(fname) as f:
            content = f.read()
            self.vectors = model.words_to_vectors(content)

    def rand_sample(self, sample_length):
        max = len(self.vectors) - sample_length - 1
        start = self.random.randint(0, max)
        return [self.vectors[i] for i in range(start, start + sample_length)], \
               [self.vectors[i] for i in range(start + 1, start + sample_length + 1)]

    def batch(self, batch_size, sample_length):
        tupled = [self.rand_sample(sample_length) for _ in range(0, batch_size)]
        return [x for x, y in tupled], [y for x, y in tupled]


def split(fname, outputs: dict):
    with open(fname) as f:
        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
        paragraphs = list(batch(f.read().split('\n'), 100))
        random.shuffle(paragraphs)
        len_paragraphs = len(paragraphs)
        paragraph_iter = iter(paragraphs)
        for subset_fname, percent in outputs.items():
            to_take = int(len_paragraphs * percent)
            subset = [paragraph for batch in list(islice(paragraph_iter, to_take)) for paragraph in batch]
            with open(subset_fname, 'w') as sub_f:
                sub_f.write('\n'.join(subset))

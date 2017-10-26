from collections import Counter
from itertools import chain
import numpy as np
from utils import pkl_utils

PAD_INDEX = 1
class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize the text.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
        vectors: A Tensor containing word vectors for the tokens in the Vocab,
            if a word vector file has been provided.
    """

    def __init__(self, corpus, dim=300):
        '''
            Arg :
                corpus: list of sentence.
        '''
        self.vectors = None
        self.vector_size = dim

        freqs = Counter(chain(*[s.split() for s in corpus]))
        self.itos = ['<unk>', '<pad>']
        self.stoi = {}

        # sort by frequency, then alphabetically
        words = sorted(freqs.items())
        words.sort(key=lambda tup: tup[1], reverse=True)

        self.itos += [s for s, v in words]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def set_vectors(self, word2index, vectors):
        assert self.vector_size == len(vectors[0])
        self.vectors = np.random.normal(size=(len(self), self.vector_size))

        new_vectors = []
        for word, index in word2index.items():
            if word in self.stoi:
                self.vectors[self.stoi[word]] = vectors[index]

    def numerize(self, text):
        return [self.stoi.get(word, 0) for word in text] 

    def save(self, path):
        pkl_utils.save(self, path)

    @classmethod
    def load(clazz, path):
        obj = pkl_utils.load(path)
        assert isinstance(obj, clazz)
        return obj


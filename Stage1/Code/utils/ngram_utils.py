# One line time: 369 ns ± 8.42 ns
def _unigrams(words):
    """
        Input: a list of words, e.g., ["I", "am", "Denny"]
        Output: a list of unigram
    """
    assert type(words) == list
    return words

# One line time: 5.04 µs ± 272 ns
def _bigrams(words, join_string='_', skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of bigram, e.g., ["I_am", "am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 1:
        lst = []
        for i in range(L-1):
            for k in range(1,skip+2):
                if i+k < L:
                    lst.append( join_string.join([words[i], words[i+k]]) )
    else:
        # set it as unigram
        lst = _unigrams(words)
    return lst

# One line time: 5.89 µs ± 196 ns
def _trigrams(words, join_string='_', skip=0):
    """
       Input: a list of words, e.g., ["I", "am", "Denny"]
       Output: a list of trigram, e.g., ["I_am_Denny"]
       I use _ as join_string for this example.
    """
    assert type(words) == list
    L = len(words)
    if L > 2:
        lst = []
        for i in range(L-2):
            for k1 in range(1,skip+2):
                for k2 in range(1,skip+2):
                    if i+k1 < L and i+k1+k2 < L:
                        lst.append(join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
    else:
        # set it as bigram
        lst = _bigrams(words, join_string, skip)
    return lst

def unichars(text):
    return _unigrams(list(text))

def bichars(text):
    return _bigrams(list(text))

def trichars(text):
    return _trigrams(list(text))

def uniwords(text):
    return _unigrams(text.split())

def biwords(text):
    return _bigrams(text.split())

def triwords(text):
    return _trigrams(text.split())


import pandas as pd
if __name__ == '__main__':

    sentence = 'My name is shawn .'
    words = sentence.split()
    x = unigrams(words)
    y = bigrams(words)
    z = trigrams(words)
    print(x)
    print(y)
    print(z)











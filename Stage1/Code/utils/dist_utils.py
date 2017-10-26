import lzma
import Levenshtein
from utils import np_utils
from scipy.spatial.distance import cosine, jaccard, cityblock, canberra, euclidean, minkowski, braycurtis

# One line time: 36.7 µs ± 388 ns
def edit_set_ratio(a, b):
    a, b = list(a), list(b)
    return Levenshtein.setratio(a, b)

# One line time: 20.7 µs ± 1.25 µs
def edit_seq_ratio(a, b):
    a, b = list(a), list(b)
    return Levenshtein.seqratio(a, b)

# One line time: 4.38 µs ± 134 ns
def jaccard_ratio(a, b):
    a, b = set(a), set(b)
    c = a & b
    return np_utils.try_divide(float(len(c)), (len(a) + len(b) - len(c)))

# One line time: 4.2 µs ± 109 ns
def dice_ratio(a, b):
    a, b = set(a), set(b)
    c = a & b
    return np_utils.try_divide(2 * float(len(c)), (len(a) + len(b)))

# One line time: 23.9 ms ± 491 µs
def lzma_ratio(a, b):
    '''Similarity after compressed using lzma'''
    if a == b:
        return 1
    a, b = a.encode('utf-8'), b.encode('utf-8')
    a_len = len(lzma.compress(a))
    b_len = len(lzma.compress(b))
    ab_len = len(lzma.compress(a + b))
    ba_len = len(lzma.compress(b + a))
    ratio = 1 - np_utils.try_divide(min(ab_len, ba_len) - min(a_len, b_len), max(a_len, b_len))
    return ratio

def cosine_distance(a, b):
    dis = np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)
    return np.nan_to_num(dis)


def jaccard_distance(a, b):
    return np.nan_to_num(jaccard(a, b))


def braycurtis_distance(a, b):
    return np.nan_to_num(braycurtis(a, b))


def canberra_distance(a, b):
    return np.nan_to_num(canberra(a, b))


def cityblock_distance(a, b):
    return np.nan_to_num(cityblock(a, b))


def euclidean_distance(a, b):
    return np.nan_to_num(euclidean(a, b))


def minkowski_distance(a, b, p=3):
    return np.nan_to_num(minkowski(a, b, p))


if __name__ == '__main__':
    test_t1 = 'asfkhioa1239asfkhkashdfkjlhaskdf13r asfnka'
    test_t2 = '123or8y908adsufinzkxnvhihsdfasifh'


import numpy as np
def lcs(seq1, seq2):
    seq1 = seq1.split()
    seq2 = seq2.split()
    dp = np.zeros((len(seq1)+1, len(seq2)+1))
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j], dp[i][j + 1])
            if seq1[i] == seq2[j]:
                dp[i + 1][j + 1] = max(dp[i][j] + 1, dp[i + 1][j + 1])
    seq1_mask = [ 0 for wd in seq1]
    seq2_mask = [ 0 for wd in seq2]
    ii, jj = len(seq1), len(seq2)
    while ii != 0 and jj != 0:
        if dp[ii][jj] == dp[ii - 1][jj - 1] + 1 and seq1[ii - 1] == seq2[jj - 1]:
            seq1_mask[ii - 1] = 1
            seq2_mask[jj - 1] = 1
            ii = ii - 1
            jj = jj - 1
            continue
        if dp[ii][jj] == dp[ii - 1][jj]:
            ii = ii - 1
        elif dp[ii][jj] == dp[ii][jj - 1]:
            jj = jj - 1
        elif dp[ii][jj] == dp[ii - 1][jj - 1]:
            ii = ii - 1
            jj = jj - 1
    seq1_left = [ wd for wd, mk in zip(seq1, seq1_mask) if mk == 0]
    seq2_left = [ wd for wd, mk in zip(seq2, seq2_mask) if mk == 0]
    
    return np.max(dp), ' '.join(seq1_left), ' '.join(seq2_left)
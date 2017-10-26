import os
import gensim
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from utils import dist_utils, ngram_utils
from scipy.stats import skew, kurtosis

import config

parser = argparse.ArgumentParser(description='Exctract the pos and ner features.')
parser.add_argument('-f', '--file', type=str, help='file name to process')
parser.add_argument('-p', '--prefix', type=str, help='prefix for features')
parser.add_argument('-d', '--save-dir', type=str, help='dir for save')
args = parser.parse_args()
if not args.file or not os.path.isfile(args.file):
    exit()
if not args.save_dir or not os.path.isdir(args.save_dir):
    exit()


def sent2vec(words):
    assert isinstance(words, list)
    vector_size = model.syn0.shape[1]
    v = np.array([np.zeros(vector_size)]+[model[w] for w in words if w in model]).mean(axis=0)
    return np.nan_to_num(v)

 
def wmd(s1, s2):
    dis = model.wmdistance(s1, s2)
    dis = np.nan_to_num(dis)
    return dis if dis<100 else 100


def norm_wmd(s1, s2):
    dis = norm_model.wmdistance(s1, s2)
    dis = np.nan_to_num(dis)
    return dis if dis<100 else 100


def run(df, ngram, prefix):
    # ngram
    df['q1_ngram'] = df['question1'].apply(ngram)
    df['q2_ngram'] = df['question2'].apply(ngram)

    # wmd
    df['%s_wmd' % prefix] = df.apply(lambda x: wmd(x['q1_ngram'], x['q2_ngram']), axis=1)
    df['%s_norm-wmd' % prefix] = df.apply(lambda x: norm_wmd(x['q1_ngram'], x['q2_ngram']), axis=1)

    # Embeddings
    ## 偏态
    df['%s_q1-skew' % prefix] = df['q1_ngram'].apply(lambda x :skew(sent2vec(x)))
    df['%s_q2-skew' % prefix] = df['q2_ngram'].apply(lambda x :skew(sent2vec(x)))
    ## 峰度
    df['%s_q1-kurtosis' % prefix] = df['q1_ngram'].apply(lambda x :kurtosis(sent2vec(x)))
    df['%s_q2-kurtosis' % prefix] = df['q2_ngram'].apply(lambda x :kurtosis(sent2vec(x)))
    ## 距离
    df['%s_cosine-distance' % prefix] = df.apply(lambda x: dist_utils.cosine_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)
    df['%s_jaccard-distance' % prefix] = df.apply(lambda x: dist_utils.jaccard_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)
    df['%s_canberra-distance' % prefix] = df.apply(lambda x: dist_utils.canberra_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)
    df['%s_cityblock-distance' % prefix] = df.apply(lambda x: dist_utils.cityblock_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)
    df['%s_euclidean-distance' % prefix] = df.apply(lambda x: dist_utils.euclidean_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)
    df['%s_minkowski-distance' % prefix] = df.apply(lambda x: dist_utils.minkowski_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)
    df['%s_braycurtis-distance' % prefix] = df.apply(lambda x: dist_utils.braycurtis_distance(sent2vec(x['q1_ngram']), sent2vec(x['q2_ngram'])), axis=1)

    df.drop(['q1_ngram', 'q2_ngram'], axis=1, inplace=True)
    return df

if __name__ == '__main__':
    global model, norm_model
    if 'glove' in args.file:
        model_file = config.GLOVE_MODEL
    elif 'word2vec' in args.file:
        model_file = config.WORD2VEC_MODEL
    else :
        exit()
    model = gensim.models.KeyedVectors.load(model_file)
    norm_model = gensim.models.KeyedVectors.load(model_file)
    norm_model.init_sims(replace=True)

    start_time = datetime.now()
    df = pd.read_csv(args.file, index_col=0)
    print('Lines: %d' %len(df))

    df = run(df, ngram_utils.uniwords, '%s_%s' % (args.prefix, 'uniwords'))

    # save
    df.drop(['question1', 'question2', 'is_duplicate'], axis=1, inplace=True, errors='ignore')
    save_path = os.path.join(args.save_dir, '%s_feature_embed.csv' % args.prefix)
    df.to_csv(save_path)
    print('Use time: {}. Save file to {}'.format(datetime.now()-start_time, save_path))




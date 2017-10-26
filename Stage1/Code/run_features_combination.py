import os
import pandas as pd
import config
import numpy as np
from datetime import datetime
from tqdm import tqdm
import gc

features_dl_files = [
    # deep learning
    'lcs_feature_dl_0.381364_0.003067.csv',
    'lemmatized_feature_dl_0.337886_0.007244.csv',
    'lemmatized_glove_feature_dl_0.339389_0.005788.csv',
    'lemmatized_glove_stemmed_feature_dl_0.344257_0.003885.csv',
    'lemmatized_stemmed_feature_dl_0.344263_0.003960.csv',
    'lemmatized_word2vec_feature_dl_0.338126_0.005606.csv',
    'lemmatized_word2vec_stemmed_feature_dl_0.345687_0.004118.csv',
    'stopword_feature_dl_0.337399_0.003806.csv',
    'tokenize_feature_dl_0.311435_0.006638.csv']

# hand craft
features_hand_files = [
    'feature_magic.csv',
    'origin_feature_david.csv',
    'lemmatized_feature_fuzz.csv',
    'lemmatized_feature_parse.csv',
    'lemmatized_feature_text.csv',
    'lemmatized_stemmed_feature_fuzz.csv',
    'lemmatized_stemmed_feature_parse.csv',
    'lemmatized_stemmed_feature_text.csv',
    'lemmatized_glove_feature_embed.csv',
    'lemmatized_glove_feature_fuzz.csv',
    'lemmatized_glove_feature_parse.csv',
    'lemmatized_glove_feature_text.csv',
    'lemmatized_glove_stemmed_feature_embed.csv',
    'lemmatized_glove_stemmed_feature_fuzz.csv',
    'lemmatized_glove_stemmed_feature_parse.csv',
    'lemmatized_glove_stemmed_feature_text.csv',
    'lemmatized_word2vec_feature_embed.csv',
    'lemmatized_word2vec_feature_fuzz.csv',
    'lemmatized_word2vec_feature_parse.csv',
    'lemmatized_word2vec_feature_text.csv',
    'lemmatized_word2vec_stemmed_feature_embed.csv',
    'lemmatized_word2vec_stemmed_feature_fuzz.csv',
    'lemmatized_word2vec_stemmed_feature_parse.csv',
    'lemmatized_word2vec_stemmed_feature_text.csv']

def combine_features(features_dir, feature_files):
    start_time = datetime.now()
    # load
    print('load feature')
    feature_files = [os.path.join(features_dir, name) for name in feature_files]
    features = pd.concat([pd.read_csv(f, index_col=0) for f in feature_files], axis=1)
    print('Use time: {}'.format(datetime.now()-start_time))
    start_time = datetime.now()

    # change inf to nan
    print('change inf to nan')
    features = features.astype(dtype=float)
    inf = np.nan_to_num(np.inf)
    features = features.replace(inf, np.nan)
    print('Use time: {}'.format(datetime.now()-start_time))
    start_time = datetime.now()

    # fill nan
    print('fill nan')
    features = features.fillna(0)
    print('Use time: {}'.format(datetime.now()-start_time))
    start_time = datetime.now()

    # symmetry the q1, q2
    print('symmetry')
    format_set = set()
    for col in tqdm(features.columns):
        col = col.replace('q1','%s')
        col = col.replace('q2','%s')
        if col not in format_set:
            format_set.add(col)
        else:
            features[col % 'diff']=(features[col % 'q1'] - features[col % 'q2']).abs()
            features[col % 'max']=np.stack([features[col % 'q1'], features[col % 'q2']]).max(0)
            features[col % 'min']=np.stack([features[col % 'q1'], features[col % 'q2']]).min(0)
        gc.collect()
    print('Use time: {}'.format(datetime.now()-start_time))
    columns = [x for x in features.columns if 'q1' not in x and 'q2' not in x]
    features = features[columns]
    gc.collect()
    return features

if __name__ == '__main__':
    # train
    start_time = datetime.now()
    train_features = combine_features(config.FEAT_TRAIN_DIR, features_hand_files)
    print('Saving file to %s' % config.FEAT_TRAIN_DATA)
    train_features.to_csv(config.FEAT_TRAIN_DATA)
    print('Use time: {}'.format(datetime.now()-start_time))

    # test
    start_time = datetime.now()
    test_features = combine_features(config.FEAT_TEST_DIR, features_hand_files)
    print('Saving file to %s' % config.FEAT_TEST_DATA)
    test_features.to_csv(config.FEAT_TEST_DATA)
    print('Use time: {}'.format(datetime.now()-start_time))





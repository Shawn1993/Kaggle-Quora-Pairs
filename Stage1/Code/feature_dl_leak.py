import os
import gensim
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from dl.vocab import Vocab
from dl import models_leak as models
import pickle
import torch
from sklearn.model_selection import StratifiedKFold
import config
import gc

parser = argparse.ArgumentParser(description='Exctract the pos and ner features.')
parser.add_argument('--train-file', type=str, help='train file name to process')
parser.add_argument('--test-file', type=str, help='test file name to process')
parser.add_argument('-p', '--prefix', type=str, help='prefix for features')
parser.add_argument('--train-save-dir', type=str, help='train dir for save')
parser.add_argument('--test-save-dir', type=str, help='test dir for save')
parser.add_argument('--dl-model', type=str, help='deep learning model name')
parser.add_argument('--wv-model', type=str, help='word2vec model')
parser.add_argument('--gpu', type=int, default=-1, help='word2vec model')
parser.add_argument('-k', '--kfold', type=int, default=8, help='number of kfold')
args = parser.parse_args()
if not args.train_file or not os.path.isfile(args.train_file):
    print('error'), exit()
if not args.test_file or not os.path.isfile(args.test_file):
    print('error'), exit()
if not args.train_save_dir or not os.path.isdir(args.train_save_dir):
    print('error'), exit()
if not args.test_save_dir or not os.path.isdir(args.test_save_dir):
    print('error'), exit()

def build_vocab(df, wv_model):
    corpus = pd.concat([df['question1'],df['question2']], ignore_index=True)
    word2index = {w: i for i, w in enumerate(wv_model.index2word)}
    vocab = Vocab(corpus)
    vocab.set_vectors(word2index, wv_model.syn0)
    return vocab

def get_train_feature(kf, data):

    # train kfold
    train_feature = []
    eval_loss = []
    for i, (train_index, dev_index) in enumerate(kf):
        dev_data = data.iloc[dev_index]
        train_data = data.iloc[train_index]
        # train
        print('Train the %dth %s model...' % (i, args.dl_model))
        model_class = getattr(models, args.dl_model)
        model = model_class(vocab, gpu=args.gpu, prefix='/tmp/%s_dl_model_cv%d.pt' % (args.prefix, i))
        loss = model.train(train_data, dev_data=dev_data)
        eval_loss.append(loss)
        # predict
        dev_data = dev_data.drop([config.COLUMN_LABEL], axis=1)
        dev_data[config.COLUMN_LABEL] = model.predict(dev_data)
        train_feature.append(dev_data[[config.COLUMN_LABEL]])
    train_feature = pd.concat(train_feature).sort_index()

    mean = np.mean(eval_loss)
    std = np.std(eval_loss)
    return train_feature, mean, std

def get_test_feature(train_data, test_data):
    print('Train the model on whole train set...')
    model_class = getattr(models, args.dl_model)
    model = model_class(vocab, gpu=args.gpu)
    model.train(train_data)
    test_data[config.COLUMN_LABEL] = model.predict(test_data)
    return test_data[[config.COLUMN_LABEL]]


if __name__ == '__main__':
    start_time = datetime.now()
    train_data = pd.read_csv(args.train_file, index_col=0).fillna('').applymap(str)
    train_leak = pd.read_csv('train_features_leak.csv', index_col=0)
    train_data = pd.concat([train_data, train_leak], axis=1)
    train_label = pd.read_csv(config.IN_TRAIN_LABEL, index_col=0)
    # train_data =next(pd.read_csv(args.train_file, index_col=0, chunksize=100)).fillna('').applymap(str)
    # train_label = next(pd.read_csv(config.IN_TRAIN_LABEL,chunksize=100, index_col=0))
    train_data[config.COLUMN_LABEL] = train_label[config.COLUMN_LABEL]
    print('Lines: %d' %len(train_data))
    #wv model
    wv_model = gensim.models.KeyedVectors.load(args.wv_model)
    vocab = build_vocab(train_data, wv_model)
    del wv_model
    gc.collect()
    train_data[config.COLUMN_Q1] = train_data[config.COLUMN_Q1].apply(lambda x : vocab.numerize(x.split()))
    train_data[config.COLUMN_Q2] = train_data[config.COLUMN_Q2].apply(lambda x : vocab.numerize(x.split()))
    # kf
    kf = pickle.load(open(config.GLOBAL_KFOLD,'rb'), encoding='bytes')
    # train feature
    train_feature, mean, std = get_train_feature(kf, train_data)
    train_feature.rename(columns={config.COLUMN_LABEL:'%s_%s' % (args.prefix, args.dl_model)}, inplace=True)
    train_save_path = os.path.join(args.train_save_dir, '%s_feature_dl_%s_%f_%f.csv' % (args.prefix, args.dl_model, mean, std))
    train_feature.to_csv(train_save_path)
    print('Save train feature to %s' % train_save_path)
    del train_feature
    gc.collect()

    # test feature
    test_data = pd.read_csv(args.test_file, index_col=0).fillna('').applymap(str)
    test_leak = pd.read_csv('test_features_leak.csv', index_col=0)
    test_data = pd.concat([test_data, test_leak], axis=1)
    # test_data = next(pd.read_csv(args.test_file, chunksize=100,index_col=0)).fillna('').applymap(str)
    test_data[config.COLUMN_Q1] = test_data[config.COLUMN_Q1].apply(lambda x : vocab.numerize(x.split()))
    test_data[config.COLUMN_Q2] = test_data[config.COLUMN_Q2].apply(lambda x : vocab.numerize(x.split()))
    test_feature = get_test_feature(train_data, test_data)
    test_feature.rename(columns={config.COLUMN_LABEL:'%s_%s' % (args.prefix, args.dl_model)}, inplace=True)
    test_save_path = os.path.join(args.test_save_dir, '%s_feature_dl_%s_%f_%f.csv' % (args.prefix, args.dl_model ,mean, std))
    test_feature.to_csv(test_save_path)
    print('Save test feature to %s' % test_save_path)
    
    print('Use time: {}.'.format(datetime.now()-start_time))




    
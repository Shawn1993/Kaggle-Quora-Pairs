import xgboost as xgb
import lightgbm as lgb
import numpy as np
import glob
import tensorflow as tf
import argparse
NTHREAD = 5


from keras import backend as K
def reset_session():
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

import pickle as pkl
import os
import sys
import numpy as np
import pandas as pd
import pickle as pkl
import glob
# hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
import datetime
# ensemble different models

from keras.models import Model, Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape,\
                        Merge, BatchNormalization, TimeDistributed,\
                        Lambda, Activation, LSTM, Flatten,\
                        Conv1D, GRU, MaxPooling1D,\
                        Conv2D, Input, MaxPooling2D, BatchNormalization, Masking

from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.optimizers import SGD, Adam, Nadam
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

def logging_output(param, train_mat, test_mat, loss, std):
    print 'logging_output'
    sys.stdout.flush()
    # date
    date = "%s" % datetime.datetime.now().strftime("[%y-%m-%d-%H_%M]")
    # data
    data_name = "[" + param['data'] + "]"
    # clf
    clf_name = []
    for key, value in param['clf'].items():
        clf_name.append("%s_%s" % (key, value))
    clf_name = "-".join(clf_name)
    clf_name = "[" + clf_name + "]"
    fn = "-".join([date, data_name, clf_name, str(loss), str(std)])
    np.save('/home/huangzhengjie/quora_pair/quora_stack/hyperopt_output/train/' + fn +'.npy', train_mat)
    np.save('/home/huangzhengjie/quora_pair/quora_stack/hyperopt_output/test/' + fn +'.npy', test_mat)



def model_train(param, X, y, X_val, y_val):
    if param['type'] == 'xgboost_gbtree':
        # go fuck yourself
        xgb_param = {}
        xgb_param['booster'] = 'gbtree'
        xgb_param['objective'] = 'binary:logistic'
        xgb_param['eval_metric'] = 'logloss'
        xgb_param['eta'] = param['eta']
        xgb_param['max_depth'] = int(param['max_depth'])
        xgb_param['subsample'] = param['subsample']
        xgb_param['colsample_bytree'] = param['colsample_bytree']
        xgb_param['nthread'] = NTHREAD
        xgb_param['gamma'] = param['gamma']
        xgb_param['min_child_weight'] = param['min_child_weight']
        xgb_param['silent'] = 1
        d_train = xgb.DMatrix(X, label=y)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        bst = xgb.train(xgb_param, d_train, int(param['num_round']), watchlist, verbose_eval=50)
        val_pred = bst.predict(d_valid)
        score = log_loss(y_val, np.array(val_pred, dtype='float64'))
        return score, val_pred
    elif param['type'] == 'nn':
        reset_session()
        model = Sequential()
        model.add(Dense(int(param['dense_dim']), activation='linear', bias=False, input_dim=X.shape[1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(param['dropout']))
        for _ in range(int(param['num_layer'])):
            model.add(Dense(int(param['dense_dim']), activation='linear', bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(param['dropout']))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        opt = Adam(lr=param['lr'])
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X, y, validation_data=(X_val, y_val), batch_size=int(param['batch_size']), nb_epoch=int(param['nb_epoch']), shuffle=True)
        val_pred = model.predict(X_val)
        score = log_loss(y_val, np.array(val_pred, dtype='float64'))
        return score, val_pred
    elif param['type'] == 'xgboost_gblinear':
        # go fuck yourself
        xgb_param = {}
        xgb_param['booster'] = 'gblinear'
        xgb_param['objective'] = 'binary:logistic'
        xgb_param['eval_metric'] = 'logloss'
        xgb_param['eta'] = param['eta']
        xgb_param['nthread'] = NTHREAD
        xgb_param['lambda'] = param['lambda']
        xgb_param['alpha'] = param['alpha']
        xgb_param['lambda_bias'] = param['lambda_bias']
        xgb_param['silent'] = 1
        d_train = xgb.DMatrix(X, label=y)
        d_valid = xgb.DMatrix(X_val, label=y_val)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        bst = xgb.train(xgb_param, d_train, int(param['num_round']), watchlist, verbose_eval=50)
        val_pred = bst.predict(d_valid)
        score = log_loss(y_val, np.array(val_pred, dtype='float64'))
        return score, val_pred
    elif param['type'] == 'logistic_regression':
        # go fuck yourself
        model = LogisticRegression(C=param['lr_C'], n_jobs=2)
        model.fit(X, y)
        print 'Testing'
        sys.stdout.flush()
        val_pred = model.predict_proba(X_val)[:, 1]
        score = log_loss(y_val, np.array(val_pred, dtype='float64'))
        return score, val_pred
    elif param['type'] == 'lgb_gbdt':
        # go fuck yourself
        xgb_param = {}
        xgb_param['boosting'] = 'gbdt',
        xgb_param['objective'] = 'binary',
        xgb_param['metric'] = 'binary_logloss'
        xgb_param['num_threads'] = NTHREAD
        xgb_param['num_leaves'] = int(param['num_leaves'])
        #xgb_param['min_data_in_leaf'] = int(param['min_data_in_leaf'])
        xgb_param['feature_fraction'] = param['feature_fraction']
        xgb_param['bagging_fraction'] = param['bagging_fraction']
        #xgb_param['bagging_freq'] = int(param['bagging_freq'])
        #xgb_param['max_bin'] = int(param['max_bin'])
        xgb_param['learning_rate'] = param['learning_rate']
        xgb_param['verbose'] = 0
        d_train = lgb.Dataset(X, y)
        d_valid = lgb.Dataset(X_val, y_val, reference=d_train)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        bst = lgb.train(xgb_param, d_train, int(param['num_iterations']), valid_sets=(d_train, d_valid), verbose_eval=5)
        val_pred = bst.predict(X_val)
        score = log_loss(y_val, np.array(val_pred, dtype='float64'))
        return score, val_pred
    elif param['type'] == 'lgb_dart':
        # go fuck yourself
        xgb_param = {}
        xgb_param['boosting'] = 'dart',
        xgb_param['objective'] = 'binary',
        xgb_param['metric'] = 'binary_logloss'
        xgb_param['num_threads'] = NTHREAD
        xgb_param['num_leaves'] = int(param['num_leaves'])
        xgb_param['skip_drop'] = param['skip_drop']
        xgb_param['drop_rate'] = param['drop_rate']
        #xgb_param['min_data_in_leaf'] = int(param['min_data_in_leaf'])
        xgb_param['feature_fraction'] = param['feature_fraction']
        xgb_param['bagging_fraction'] = param['bagging_fraction']
        #xgb_param['bagging_freq'] = int(param['bagging_freq'])
        #xgb_param['max_bin'] = int(param['max_bin'])
        xgb_param['learning_rate'] = param['learning_rate']
        xgb_param['verbose'] = 0
        d_train = lgb.Dataset(X, y)
        d_valid = lgb.Dataset(X_val, y_val, reference=d_train)
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        bst = lgb.train(xgb_param, d_train, int(param['num_iterations']), valid_sets=(d_train, d_valid), verbose_eval=5)
        val_pred = bst.predict(X_val)
        score = log_loss(y_val, np.array(val_pred, dtype='float64'))
        return score, val_pred



def model_test(param, X, y, X_test):
    if param['type'] == 'xgboost_gbtree':
        # go fuck yourself
        xgb_param = {}
        xgb_param['booster'] = 'gbtree'
        xgb_param['objective'] = 'binary:logistic'
        xgb_param['eval_metric'] = 'logloss'
        xgb_param['eta'] = param['eta']
        xgb_param['max_depth'] = int(param['max_depth'])
        xgb_param['subsample'] = param['subsample']
        xgb_param['colsample_bytree'] = param['colsample_bytree']
        xgb_param['nthread'] = NTHREAD
        xgb_param['gamma'] = param['gamma']
        xgb_param['min_child_weight'] = param['min_child_weight']
        xgb_param['silent'] = 1

        d_train = xgb.DMatrix(X, label=y)
        d_test = xgb.DMatrix(X_test)
        watchlist = [(d_train, 'train')]
        bst = xgb.train(xgb_param, d_train, int(param['num_round']), watchlist, verbose_eval=50)
        print 'Testing'
        sys.stdout.flush()
        pred = bst.predict(d_test)
        print 'Testing Finish'
        sys.stdout.flush()
        return pred
    elif param['type'] == 'nn':
        reset_session()
        model = Sequential()
        model.add(Dense(int(param['dense_dim']), activation='linear', bias=False, input_dim=X.shape[1]))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(param['dropout']))
        for _ in range(int(param['num_layer'])):
            model.add(Dense(int(param['dense_dim']), activation='linear', bias=False))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(param['dropout']))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        opt = Adam(lr=param['lr'])
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(X, y, batch_size=int(param['batch_size']), nb_epoch=int(param['nb_epoch']), shuffle=True)
        pred = model.predict(X_test)
        return pred
    elif param['type'] == 'xgboost_gblinear':
        # go fuck yourself
        xgb_param = {}
        xgb_param['booster'] = 'gblinear'
        xgb_param['objective'] = 'binary:logistic'
        xgb_param['eval_metric'] = 'logloss'
        xgb_param['lambda'] = param['lambda']
        xgb_param['alpha'] = param['alpha']
        xgb_param['lambda_bias'] = param['lambda_bias']
        xgb_param['eta'] = param['eta']
        xgb_param['nthread'] = NTHREAD
        xgb_param['silent'] = 1

        d_train = xgb.DMatrix(X, label=y)
        d_test = xgb.DMatrix(X_test)
        watchlist = [(d_train, 'train')]
        bst = xgb.train(xgb_param, d_train, int(param['num_round']), watchlist, verbose_eval=50)
        print 'Testing'
        sys.stdout.flush()
        pred = bst.predict(d_test)
        print 'Testing Finish'
        sys.stdout.flush()
        return pred
    elif param['type'] == 'logistic_regression':
        # go fuck yourself
        model = LogisticRegression(C=param['lr_C'], n_jobs=2)
        model.fit(X, y)
        print 'Testing'
        sys.stdout.flush()
        pred = model.predict_proba(X_test)[:, 1]
        print 'Testing Finish'
        sys.stdout.flush()
        return pred
    elif param['type'] == 'lgb_gbdt':
        # go fuck yourself
        xgb_param = {}
        xgb_param['boosting'] = 'gbdt',
        xgb_param['objective'] = 'binary',
        xgb_param['metric'] = 'binary_logloss'
        xgb_param['num_threads'] = NTHREAD
        xgb_param['learning_rate'] = param['learning_rate']
        xgb_param['num_leaves'] = int(param['num_leaves'])
        #xgb_param['min_data_in_leaf'] = int(param['min_data_in_leaf'])
        xgb_param['feature_fraction'] = param['feature_fraction']
        xgb_param['bagging_fraction'] = param['bagging_fraction']
        #xgb_param['bagging_freq'] = int(param['bagging_freq'])
        #xgb_param['max_bin'] = int(param['max_bin'])
        xgb_param['verbose'] = 0

        d_train = lgb.Dataset(X, y)
        d_test = lgb.Dataset(X_test)
        bst = lgb.train(xgb_param, d_train, int(param['num_iterations']), valid_sets=d_train, verbose_eval=50)
        print 'Testing'
        sys.stdout.flush()
        pred = bst.predict(X_test)
        print 'Testing Finish'
        sys.stdout.flush()
        return pred
    elif param['type'] == 'lgb_dart':
        # go fuck yourself
        xgb_param = {}
        xgb_param['boosting'] = 'dart',
        xgb_param['objective'] = 'binary',
        xgb_param['metric'] = 'binary_logloss'
        xgb_param['num_threads'] = NTHREAD
        xgb_param['learning_rate'] = param['learning_rate']
        xgb_param['num_leaves'] = int(param['num_leaves'])
        #xgb_param['min_data_in_leaf'] = int(param['min_data_in_leaf'])
        xgb_param['feature_fraction'] = param['feature_fraction']
        xgb_param['bagging_fraction'] = param['bagging_fraction']
        xgb_param['skip_drop'] = param['skip_drop']
        xgb_param['drop_rate'] = param['drop_rate']
        #xgb_param['bagging_freq'] = int(param['bagging_freq'])
        #xgb_param['max_bin'] = int(param['max_bin'])
        xgb_param['verbose'] = 0

        d_train = lgb.Dataset(X, y)
        d_test = lgb.Dataset(X_test)
        bst = lgb.train(xgb_param, d_train, int(param['num_iterations']), valid_sets=d_train, verbose_eval=50)
        print 'Testing'
        sys.stdout.flush()
        pred = bst.predict(X_test)
        print 'Testing Finish'
        sys.stdout.flush()
        return pred







y_train = np.load("/home/huangzhengjie/quora_pair/quora_stack/label.npy", mmap_mode="r")
with open("/home/huangzhengjie/quora_pair/quora_stack/kdf.pkl", 'r') as f:
    kfd = pkl.load(f)
def hyperopt_wrapper(param):
    print param
    if param['data'] == 'zj':
        x_train = np.load("/data1/huangzhengjie/hyperopt/zj_train.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/zj_test.npy", mmap_mode='r')
    elif param['data'] == 'wxh':
        x_train = np.load("/data1/huangzhengjie/hyperopt/train_wxh.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/test_wxh.npy", mmap_mode='r')
    elif param['data'] == 'zjb':
        x_train = np.load("/data1/huangzhengjie/hyperopt/train_zjb.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/test_zjb.npy", mmap_mode='r')
    elif param['data'] == 'bst':
        x_train = np.load("/data1/huangzhengjie/hyperopt/train_bst.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/test_bst.npy", mmap_mode='r')
    elif param['data'] == 'excalibur':
        x_train = np.load("/data1/huangzhengjie/hyperopt/train_excalibur.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/test_excalibur.npy", mmap_mode='r')
    elif param['data'] == 'excalibur_v2':
        x_train = np.load("/data1/huangzhengjie/hyperopt/train_excalibur_v2.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/test_excalibur_v2.npy", mmap_mode='r')
    elif param['data'] == 'excalibur_v5':
        x_train = np.load("/data1/huangzhengjie/hyperopt/train_excalibur_v5.npy", mmap_mode='r')
        x_test = np.load("/data1/huangzhengjie/hyperopt/test_excalibur_v5.npy", mmap_mode='r')
    xgb_train_output = np.zeros(len(x_train))
    log_score = []
    for fold_id, (train_index, test_index) in enumerate(kfd):
        score, val_pred = model_train(param['clf'], x_train[train_index], y_train[train_index], x_train[test_index], y_train[test_index])
        xgb_train_output[test_index] = val_pred
        log_score.append(score)
        print 'Fold_id', fold_id, 'log_loss', score
        sys.stdout.flush()
    prediction = model_test(param['clf'], x_train, y_train, x_test)

    logging_output(param, xgb_train_output, prediction, np.mean(log_score), np.std(log_score))
    return np.mean(score)

# hyperopt tuning space
xgb_gbtree_param = {
    'type': 'xgboost_gbtree',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'max_depth': hp.quniform('max_depth', 4, 6, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'num_round': hp.quniform('num_round', 50, 200, 5),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'gamma': hp.quniform('gamma', 0, 2, 0.1),
}

# hyperopt tuning space
lr_space = {
    'type': 'logistic_regression',
    'lr_C': hp.loguniform("lr_C", np.log(0.001), np.log(10)),
}

# hyperopt tuning space
xgb_gblinear_param = {
    'type': 'xgboost_gblinear',
    'eta': hp.quniform('eta', 0.01, 1, 0.01),
    'max_depth': hp.quniform('max_depth', 4, 6, 1),
    'num_round': hp.quniform('num_round', 50, 200, 5),
    'lambda' : hp.quniform('lambda', 0, 5, 0.05),
    'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
    'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
}



nn_space = {
    'type': 'nn',
    'lr': hp.quniform("lr", 0.001, 0.1, 0.001),
    'dropout': hp.quniform("dropout", 0.3, 0.6, 0.1),
    'nb_epoch': hp.quniform('nb_epoch', 20, 30, 5),
    'batch_size': hp.choice('batch_size', [512, 1024, 2048]),
    'num_layer': hp.choice("num_layer", [1, 2, 3]),
    'dense_dim': hp.quniform("dense_dim", 32, 128, 5)
}



lightgbm_gbdt_space = {
    'type': 'lgb_gbdt',
    'num_leaves': hp.quniform("num_leaves", 50, 100, 5),
    #"min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 40, 5),
    "feature_fraction": hp.quniform("feature_fraction", 0.7, 1.0, 0.1),
    "bagging_fraction": hp.quniform("bagging_fraction", 0.7, 1.0, 0.1),
    #"bagging_freq": hp.quniform("bagging_freq", 0, 50, 10),
    "max_bin": 255,
    "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
    'num_iterations': hp.quniform("num_iterations", 100, 300, 5)
}

lightgbm_dart_space = {
    'type': 'lgb_dart',
    'drop_rate': hp.quniform('drop_rate', 0.5, 1.0, 0.1),
    'skip_drop': hp.quniform('skip_drop', 0.5, 1.0, 0.1),
    'num_leaves': hp.quniform("num_leaves", 50, 100, 5),
    #"min_data_in_leaf": hp.quniform("min_data_in_leaf", 20, 40, 5),
    "feature_fraction": hp.quniform("feature_fraction", 0.7, 1.0, 0.1),
    "bagging_fraction": hp.quniform("bagging_fraction", 0.7, 1.0, 0.1),
    #"bagging_freq": hp.quniform("bagging_freq", 0, 50, 10),
    "max_bin": 255,
    "learning_rate": hp.quniform("learning_rate", 0.01, 0.2, 0.01),
    'num_iterations': hp.quniform("num_iterations", 100, 300, 5)
}

'''
param_space = {
    'data': hp.choice('data', ['zjb']),
    'clf': hp.choice('clf', [xgboost_space])
}
'''

def parse_args():
    description = '''
    data [ zj, zjb, wxh ]
    clf [xgb, nn, randomforest]
    '''
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('data', type=str)
    parser.add_argument('clf', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    param_space = {
    }
    param_space['data'] = args.data
    if args.clf == 'xgb_tree':
        param_space['clf'] = xgb_gbtree_param
        max_evals = 50
    elif args.clf == 'xgb_linear':
        param_space['clf'] = xgb_gblinear_param
        max_evals = 50
    elif args.clf == 'nn':
        param_space['clf'] = nn_space
        max_evals = 50
    elif args.clf == 'lr':
        param_space['clf'] = lr_space
        max_evals = 20
    elif args.clf == 'lgb_gbdt':
        param_space['clf'] = lightgbm_gbdt_space
        max_evals = 50
    elif args.clf == 'lgb_dart':
        param_space['clf'] = lightgbm_dart_space
        max_evals = 50
    best_params = fmin(hyperopt_wrapper, param_space, algo=tpe.suggest, trials=Trials(), max_evals=max_evals)
    print best_params

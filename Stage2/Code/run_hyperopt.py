import os
import config
import pickle
import numpy as np
import pandas as pd
import time
import json
import log_utils
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss
from models import LigthGBM_DART, LigthGBM_GDBT, Sklearn_RF, Torch_DNN
model_map = {
    'lgb_dart': LigthGBM_DART,
    'lgb_gbdt': LigthGBM_GDBT,
    'sklearn_rf': Sklearn_RF,
    'torch_nn': Torch_DNN
}

@log_utils.usetime('Load kfold')
def _load_kfold():
    kf = pickle.load(open(config.KFOLD_FILE, 'rb'), encoding='bytes')
    return kf

@log_utils.usetime('Load train data')
def _load_train():
    X = np.load('train_excalibur_v5.npy', mmap_mode='r')
    y = np.load(config.IN_TRAIN_LABEL, mmap_mode='r')
    return X, y

@log_utils.usetime('Load test data')
def _load_test():
    X = np.load('test_excalibur_v5.npy', mmap_mode='r')
    return X

kf = _load_kfold()
X, y =  _load_train()
test_X = _load_test()

@ log_utils.usetime('Optimizing')
def fn(params):
    model_name = params.pop('model')
    model = model_map[model_name](**params)
    
    print(json.dumps(params, indent=1))

    train_predicts = np.empty_like(y)
    losses = []
    for i, (train_index, valid_index) in enumerate(kf):
        train_X, train_y = X[train_index], y[train_index]
        valid_X, valid_y = X[valid_index], y[valid_index]
        eval_loss = model.fit(train_X, train_y, valid_X, valid_y)
        losses.append(eval_loss)
        train_predicts[valid_index] = model.predict(valid_X)

    timestamp, mean, std = int(time.time()), np.mean(losses), np.std(losses)
    filename = '%s_%d_%f_%f_[parmas]_%s.npy' % (model_name, timestamp, mean, std, json.dumps(params))
    np.save(os.path.join(config.OUTPUT_TRAIN_DIR, filename), train_predicts)
    # Save Test
    model.fit(X, y)
    test_predicts = model.predict(test_X)
    np.save(os.path.join(config.OUTPUT_TEST_DIR, filename), test_predicts)

    return {
        'loss': mean,
        'loss_variance': std,
        'status': STATUS_OK,
        'file_name': filename
    }

def run_lgb_dart():
    trials = Trials()
    search_space = {
        'model': 'lgb_dart',
        'num_threads': 2, 
        'drop_rate': hp.quniform('drop_rate', 0.5, 1, 0.1),
        'skip_drop': hp.quniform('skip_drop', 0.5, 1, 0.1),
        'num_leaves': hp.quniform('num_leaves', 15, 100, 5),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 100, 10),
        'feature_fraction': hp.quniform('feature_fraction', 0.5, 1.0, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.5, 1.0, 0.1),
        'bagging_freq': hp.quniform('bagging_freq', 0, 50, 10),
        'max_bin': hp.choice('max_bin', [15, 63, 255, 1023]),
        'learning_rate': hp.quniform('learning_rate', 0.01, 1, 0.01),
        'num_iterations': hp.quniform('num_iterations', 50, 500, 10)
    }
    best_param = fmin(fn,
                      space=search_space,
                      algo=tpe.suggest,
                      max_evals=100,
                      trials=trials)


    info = trials.best_trial
    info['param'] = best_param

    json.dump(info, open('lgb_dart_best_info.json','w'),indent=1)    

def run_lgb_gbdt():
    trials = Trials()
    search_space = {
        'model': 'lgb_gbdt',
        'num_threads':8,
        'num_leaves': hp.quniform('num_leaves', 20, 100, 5),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 10, 100, 10),
        'feature_fraction': hp.quniform('feature_fraction', 0.6, 1.0, 0.1),
        'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 1.0, 0.1),
        'bagging_freq': hp.quniform('bagging_freq', 0, 20, 2),
        'max_bin': hp.choice('max_bin', [15, 63, 255, 1023]),
        'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.01),
        'num_iterations': hp.quniform('num_iterations', 100, 500, 50)
    }
    best_param = fmin(fn,
                      space=search_space,
                      algo=tpe.suggest,
                      max_evals=100,
                      trials=trials)


    info = trials.best_trial
    info['param'] = best_param

    json.dump(info, open('lgb_gbdt_best_info.json','w'),indent=1)

def run_sklearn_rf():
    trials = Trials()
    search_space = {
        'model': 'sklearn_rf',
        'n_estimators': hp.quniform('n_estimators', 100, 500, 20),
        'max_features': hp.quniform('max_features', 0.05, 1.0, 0.05),
        'n_jobs': 8
    }

    best_param = fmin(fn,
              space=search_space,
              algo=tpe.suggest,
              max_evals=200,
              trials=trials)


    info = trials.best_trial
    info['param'] = best_param
    json.dump(info, open('rf_best_info.json','w'),indent=1)

def run_torch_nn():
    trials = Trials()
    search_space = {
        'model': 'torch_nn',
        'gpu': 3,
        'batch_size' : 64,
        'max_iters': hp.quniform('max_iters', 1, 3, 1),
        'H1': hp.quniform('H1', 64, 256, 8),
        'H2': hp.quniform('H2', 64, 256, 8),
        'dropout': hp.quniform('dropout', 0, 0.5, 0.01),
    }

    best_param = fmin(fn,
              space=search_space,
              algo=tpe.suggest,
              max_evals=200,
              trials=trials)


    info = trials.best_trial
    info['param'] = best_param
    json.dump(info, open('rf_best_info.json','w'), indent=1)



import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()
    if args.model == 'gbdt':
        run_lgb_gbdt()
    elif args.model == 'dart':
        run_lgb_dart()
    elif args.model == 'rf':
        run_sklearn_rf()
    elif args.model == 'nn':
        run_torch_nn()

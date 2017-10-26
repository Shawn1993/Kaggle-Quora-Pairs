import os
import config
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

def get_train_features(kf, train_features, train_label):
    ## hyper parameter
    HYPER_PARAM = {
        'num_leaves': 50,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8
    }
    NUM_ROUND = 200000
    EARLY_STOP = 50
    K_FOLD = 10

    ## cross validation
    train_feature = []
    eval_loss = []
    best_iter = []
    for i, (train_index, dev_index) in enumerate(kf):
        train_data = lgb.Dataset(train_features.iloc[train_index], label=train_label[train_index])
        dev_data = lgb.Dataset(train_features.iloc[dev_index], label=train_label[dev_index])
        
        print('Training the %dth model...' % i)
        start_time = datetime.now()
        bst = lgb.train(HYPER_PARAM,
                        train_data,
                        verbose_eval=10,
                        valid_sets=[train_data, dev_data],
                        valid_names=['Train', 'Dev'],
                        num_boost_round=NUM_ROUND,
                        early_stopping_rounds=EARLY_STOP)
        print('Use time: {}'.format(datetime.now()-start_time))
        best_iter.append(bst.best_iteration)
        eval_loss.append(bst.best_score['Dev']['binary_logloss'])
        train_feature.append(pd.DataFrame(index=dev_index, data={'is_duplicate': bst.predict(train_features.iloc[dev_index])}))
    train_feature = pd.concat(train_feature).sort_index()

    mean = np.mean(eval_loss)
    std = np.std(eval_loss)
    best_iter = int(np.mean(best_iter))
    return train_feature, mean, std, best_iter

def get_test_features(train_features, train_label, test_features, best_iter):
    ## hyper parameter
    HYPER_PARAM = {
        'num_leaves': 50,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'feature_fraction' : 0.8,
        'bagging_fraction' : 0.8
    }
    NUM_ROUND = best_iter
    # train
    print('Training the whole model...')
    start_time = datetime.now()
    train_data = lgb.Dataset(train_features, label=train_label)
    bst = lgb.train(HYPER_PARAM,
                    train_data,
                    valid_sets=[train_data],
                    verbose_eval=10,
                    num_boost_round=NUM_ROUND)
    print('Use time: {}'.format(datetime.now()-start_time))
    # test
    return pd.DataFrame(index=test_features.index, data={'is_duplicate': bst.predict(test_features)})

if __name__ == '__main__':
    start_time = datetime.now()
    features_importance = pd.read_csv(config.IN_FEAT_IMPORTANCE, index_col=0)
    columns = features_importance[:160].index.tolist()
    # train
    print('Load train features')
    # train_features_dl = next(pd.read_csv(config.IN_TRAIN_FEAT_DL, index_col=0, chunksize=100))
    # train_features_hand = next(pd.read_csv(config.IN_TRAIN_FEAT_HAND, index_col=0, usecols=['id'] + columns, chunksize=100))
    # train_label = next(pd.read_csv(config.IN_TRAIN_LABEL, index_col=0,  chunksize=100))['is_duplicate'].values
    train_features_dl = pd.read_csv(config.IN_TRAIN_FEAT_DL, index_col=0)
    train_features_hand = pd.read_csv(config.IN_TRAIN_FEAT_HAND, index_col=0, usecols=['id'] + columns)
    train_features = pd.concat([train_features_dl, train_features_hand], axis=1)
    train_label = pd.read_csv(config.IN_TRAIN_LABEL, index_col=0)['is_duplicate'].values
    
    kf = pickle.load(open(config.KFOLD_FILE, 'rb'), encoding='bytes')

    train_output, mean, std, best_iter = get_train_features(kf, train_features, train_label)
    train_output.rename(columns={config.COLUMN_LABEL:'lgb_gbdt'}, inplace=True)
    train_save_path = os.path.join(config.OUTPUT_TRAIN_DIR, 'lgb_gbdt_%d_%f_%f.csv' % (best_iter, mean, std))
    train_output.to_csv(train_save_path)
    print('Save train feature to %s' % train_save_path)

    # test
    print('Load test features')
    # test_features_dl = next(pd.read_csv(config.IN_TEST_FEAT_DL, index_col=0, chunksize=100))
    # test_features_hand = next(pd.read_csv(config.IN_TEST_FEAT_HAND, index_col=0, usecols=['test_id'] + columns, chunksize=100))

    test_features_dl = pd.read_csv(config.IN_TEST_FEAT_DL, index_col=0)
    test_features_hand = pd.read_csv(config.IN_TEST_FEAT_HAND, index_col=0, usecols=['test_id'] + columns)
    test_features = pd.concat([test_features_dl, test_features_hand], axis=1)
    test_output = get_test_features(train_features, train_label, test_features, best_iter)
    test_output.rename(columns={config.COLUMN_LABEL:'lgb_gbdt'}, inplace=True)
    test_save_path = os.path.join(config.OUTPUT_TEST_DIR, 'lgb_gbdt_%d_%f_%f.csv' % (best_iter, mean, std))
    test_output.to_csv(test_save_path)
    print('Save test feature to %s' % test_save_path)
    print('Use time: {}.'.format(datetime.now()-start_time))


import os
import config
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from minepy import MINE
from utils import log_utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RandomizedLasso


def normalize(ranks, order=1):
    ranks = np.array(ranks, dtype=float)
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*ranks.reshape([-1, 1])).reshape(-1)
    ranks = np.round(ranks, 3)
    return ranks


@log_utils.usetime('Linear Regression')
def linear_regression(X, y):
    lr = LinearRegression(normalize=True).fit(X, y)
    return normalize(np.abs(lr.coef_))


@log_utils.usetime('Ridge')
def ridge(X, y):
    ridge = Ridge(alpha=2).fit(X, y)
    return normalize(np.abs(ridge.coef_))


@log_utils.usetime('Lasso')
def lasso(X, y):
    lasso = Lasso(alpha=0.05).fit(X, y)
    return normalize(np.abs(lasso.coef_))


@log_utils.usetime('Stability Selection')
def stability_select(X, y):
    rlasso = RandomizedLasso(alpha=0.04).fit(X, y)
    return normalize(np.abs(rlasso.scores_))


@log_utils.usetime('REF')
def ref(X, y):
    rfe = RFE(LinearRegression(normalize=True),
              n_features_to_select=50).fit(X, y)
    return normalize(rfe.ranking_, order=-1)


@log_utils.usetime('Correlation')
def correlation(X, y):
    f, pval = f_regression(X, y, center=True)
    return normalize(f)

@log_utils.usetime('Random Forest')
def random_forest(X, y):
    rf = RandomForestRegressor().fit(X, y)
    return normalize(rf.feature_importances_)


@log_utils.usetime('LightGBM')
def light_gbm(X, y):
    HYPER_PARAM = {
        'num_leaves': 50,
        'objective': 'binary',
        'metric': 'binary_logloss',
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'silent': 1
    }
    NUM_ROUND = 1000
    EARLY_STOP = 50
    skf = StratifiedKFold(8, shuffle=True)
    importances = []
    for i, (train_index, dev_index) in enumerate(skf.split(X, y)):
        train_data = lgb.Dataset(X[train_index], label=y[train_index])
        dev_data = lgb.Dataset(X[dev_index], label=y[dev_index])

        bst = lgb.train(HYPER_PARAM,
                        train_data,
                        valid_sets=[dev_data],
                        valid_names=['Dev'],
                        num_boost_round=NUM_ROUND,
                        early_stopping_rounds=EARLY_STOP)
        importances.append(bst.feature_importance())
    importances = np.stack(importances).mean(0)
    return normalize(importances)


def run():
    parser = argparse.ArgumentParser('Use to feature selcection')
    parser.add_argument('--method', type=str, help='use which method')
    args = parser.parse_args()

    np.random.seed(config.RANDOM_STATE)
    # load data
    print('Load data')
    train_features = pd.read_csv(config.TRAIN_FEAT_HAND, index_col=0)
    train_labels = pd.read_csv(config.TRAIN_LABEL, index_col=0)
    X = train_features.values
    y = train_labels.values.reshape([-1])

    ranks = pd.DataFrame(index=train_features.columns)

    if args.method == 'simpile':
        ranks['Ridge'] = ridge(X, y)
        ranks['Lasso'] = lasso(X, y)
        ranks['Correlation'] = correlation(X, y)
        ranks['Linear Regression'] = linear_regression(X, y)
        ranks['Stability Selection'] = stability_select(X, y)
    elif args.method == 'ref':
        ranks['REF'] = ref(X, y)
    elif args.method == 'lgb':
        ranks['LightGBM'] = light_gbm(X, y)
    elif args.method == 'rf':
        ranks['Random Forest'] = random_forest(X, y)
    elif args.method == 'mic':
        ranks['MIC'] = mic(X, y)
    else:
        print('No such method !')

    print('Save to features_weight_%s.csv' % args.method)
    ranks.to_csv('features_weight_%s.csv' % args.method)

if __name__ == '__main__':
    python = 'python3'

    cmd = '%s run_features_selcation.py --method=simple &' % (python)
    print('[Run command]:\n' + cmd)
    os.system(cmd)

    cmd = '%s run_features_selcation.py --method=rfe &' % (python)
    print('[Run command]:\n' + cmd)
    os.system(cmd)

    cmd = '%s run_features_selcation.py --method=rf &' % (python)
    print('[Run command]:\n' + cmd)
    os.system(cmd)

    cmd = '%s run_features_selcation.py --method=lgb &' % (python)
    print('[Run command]:\n' + cmd)
    os.system(cmd)

    # cmd = '%s run_features_selcation.py --method=mic &' %(python)
    # print('[Run command]:\n' + cmd)
    # os.system(cmd)

import log_utils
try:
    import lightgbm as lgb
except Exception as e:
    pass
import sys
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss
class LigthGBM(object):

    def __init__(self, num_leaves, min_data_in_leaf, max_bin, feature_fraction, bagging_fraction, bagging_freq, num_iterations, learning_rate, num_threads):
        self.HYPER_PARAM = {
            'num_threads':num_threads,
            'verbose':0,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves':int(num_leaves),
            'min_data_in_leaf':int(min_data_in_leaf),
            'feature_fraction':feature_fraction,
            'bagging_fraction':bagging_fraction,
            'bagging_freq':int(bagging_freq),
            'max_bin':int(max_bin),
            'learning_rate':learning_rate
        }
        self.num_boost_round = int(num_iterations)

    @log_utils.usetime('fit')
    def fit(self, X, y, valid_X=None, valid_y=None):
        evals_result = {}
        train_set = lgb.Dataset(X, label=y)
        if valid_X is None or valid_y is None:
            self.bst = lgb.train(self.HYPER_PARAM,
                            train_set,
                            verbose_eval=50,
                            num_boost_round=self.num_boost_round)
        else :
            valid_set = lgb.Dataset(valid_X, label=valid_y)
            self.bst = lgb.train(self.HYPER_PARAM,
                            train_set,
                            verbose_eval=50,
                            evals_result = evals_result,
                            valid_names=['valid'],
                            valid_sets=[valid_set],
                            num_boost_round=self.num_boost_round)
            self.valid_loss = evals_result['valid']['binary_logloss']

    def predict(self, X):
        return self.bst.predict(X)

class LigthGBM_DART(LigthGBM):
    def __init__(self, drop_rate, skip_drop, **kargv):
        super(LigthGBM_DART, self).__init__(**kargv)
        self.HYPER_PARAM.update({
            'drop_rate': drop_rate,
            'skip_drop': skip_drop,
            'boosting':'dart'
        })

class LigthGBM_GDBT(LigthGBM):
    def __init__(self, **kargv):
        super(LigthGBM_GDBT, self).__init__(**kargv)
        self.HYPER_PARAM.update({
            'boosting':'gbdt'
        })

class Sklearn_RF(object):
    """docstring for RandomForest"""
    def __init__(self, n_estimators, min_samples_split, min_samples_leaf,min_weight_fraction_leaf, max_leaf_nodes, n_jobs=-1):
        super(Sklearn_RF, self).__init__()
        self.HYPER_PARAM = {
            'n_estimators':int(n_estimators),
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_leaf_nodes':int(max_leaf_nodes),
            'n_jobs': n_jobs
        }
    def fit(self, X, y, valid_X=None, valid_y=None):
        self.rfc = RandomForestClassifier(**self.HYPER_PARAM, verbose=10).fit(X, y)
        if valid_X is not None and valid_y is not None:
            self.valid_loss = log_loss(valid_y, self.rfc.predict_proba(valid_X))

    def predict(self, X):
        return self.rfc.predict_proba(X)[:, 1]


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class Torch_DNN(nn.Module):
    """docstring for RandomForest"""
    def __init__(self, max_iters=2, batch_size=64, H1=100, H2=100, dropout=0.5, gpu=-1):
        super(Torch_DNN, self).__init__()
        self.max_iters = int(max_iters)
        self.batch_size = int(batch_size)
        self.dropout = dropout
        self.gpu = int(gpu)
        self.H1 = int(H1)
        self.H2 = int(H2)

        if gpu >= 0:
            self.LongTensor = torch.cuda.LongTensor
            self.FloatTensor = torch.cuda.FloatTensor
        else :
            self.LongTensor = torch.LongTensor
            self.FloatTensor = torch.FloatTensor

    def _init_modules(self, features_size):
        self.fc1 = nn.Linear(features_size, self.H1)
        self.fc2 = nn.Linear(self.H1, self.H2)
        self.output = nn.Linear(self.H2, 2)

    def forward(self, x, training):
        x = F.dropout(self.fc1(x), training=training, p=self.dropout)
        x = F.dropout(self.fc2(x), training=training, p=self.dropout)
        return  self.output(x)

    def _load_batch(self, X, y=None):
        training = (y is not None)
        data_size = len(X)
        indices = np.random.permutation(data_size)
        for i in range(0, data_size, self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            if training:
                yield Variable(torch.from_numpy(X[batch_indices]).type(self.FloatTensor), volatile=not training),\
                       Variable(torch.from_numpy(y[batch_indices]).type(self.LongTensor), volatile=not training)
            else :
                yield Variable(torch.from_numpy(X[batch_indices]).type(self.FloatTensor), volatile=not training)

    def _loss(self, logits, y_true):
        return F.cross_entropy(logits, y_true.view(-1).long())
    
    def _eval(self, logits, y_true):
        evaluates = (F.softmax(logits).max(1)[1].long() == y_true.long()).long()
        return torch.sum(evaluates)

    def _predict(self, logits):
        return F.softmax(logits)[:, 1:].data.cpu().numpy()

    def fit(self, X, y, valid_X=None, valid_y=None):
        self._init_modules(X.shape[1])
        if self.gpu >= 0 :
            torch.cuda.set_device(self.gpu)
            self.cuda()
        optimizer = torch.optim.Adam(self.parameters())
        steps = 1
        is_eval = (valid_X is not None and valid_y is not None)
        for itr in range(self.max_iters):
            start_time = datetime.now()
            all_loss = 0
            data_size = 0
            for batch_X, batch_y in self._load_batch(X, y):

                logits = self(batch_X, training=True)
                loss = self._loss(logits, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # for logging
                sampel_size = batch_X.size(0)
                data_size += sampel_size
                all_loss += loss.data[0] * sampel_size
                steps += 1
                sys.stdout.write('\rIter[{}] - Step[{}] - Loss:{:.6f} - Timer:{}'.format(itr + 1,
                                                                                         steps,
                                                                                         loss.data[0],
                                                                                         datetime.now() - start_time))
                # if steps % 400 == 0 and itr>=0 and is_eval:
                #     mean_loss = all_loss / data_size
                #     sys.stdout.write('\rIter[{}] - Step[{}] - Mean Loss:{:.6f} - Timer:{}\n'.format(itr + 1,
                #                                                                              steps,
                #                                                                              mean_loss,
                #                                                                              datetime.now() - start_time))
         
                #     self.eval(valid_X, valid_y, 'Dev')

        if is_eval:
            return self.eval(valid_X, valid_y, 'Dev')

    def eval(self, X, y, name):
        all_loss = 0
        all_correct = 0
        for batch_X, batch_y in self._load_batch(X, y):
            logits = self(batch_X, training=False)
            loss = self._loss(logits, batch_y)
            correct = self._eval(logits, batch_y)

            all_loss += loss.data[0] * len(batch_y)
            all_correct += correct.data[0]

        data_size = X.shape[0]
        mean_loss = all_loss / data_size
        mean_correct = all_correct / data_size

        sys.stdout.write('[{}] - Acc:{:.4f}({}/{}) - Loss:{:.6f}\n'.format(name,
                                                                           mean_correct,
                                                                           all_correct,
                                                                           data_size,
                                                                           mean_loss))
        return mean_loss

    def predict(self, X):
        if self.gpu >= 0 :
            torch.cuda.set_device(self.gpu)
            self.cuda()
        all_predicts = np.empty([0, 1])
        for batch_X in self._load_batch(X):
            cur_predicts = self._predict(self(batch_X, training=False))
            all_predicts = np.concatenate([all_predicts, cur_predicts])
        return all_predicts

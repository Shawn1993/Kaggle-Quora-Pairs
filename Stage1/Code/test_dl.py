import pandas as pd
from sklearn.model_selection import train_test_split
from utils import log_utils
import config
import gensim
from dl.vocab import Vocab
from dl.models_leak import SelfAttentionBiLSTM, SimpleCNN

@log_utils.usetime('Building Vocabulary...')
def build_vocab(df, wv_model):
    corpus = pd.concat([
        df[config.COLUMN_Q1],
        df[config.COLUMN_Q2]
    ], ignore_index=True)
    word2index = {w: i for i, w in enumerate(wv_model.index2word)}
    vocab = Vocab(corpus)
    vocab.set_vectors(word2index, wv_model.syn0)
    return vocab


@log_utils.usetime('Training the deep learning model...')
def fit(vocab, train_data, dev_data):
    # train
    model = SimpleCNN(vocab, gpu=0)
    model.train(train_data, dev_data=dev_data)

if __name__ == '__main__':

    # load_data
    train_data = pd.read_csv(config.IN_TRAIN_DIR+'/lower_rmpunctuation_tokenize.csv', index_col=0).fillna('').applymap(str)
    train_leak = pd.read_csv(config.OUT_TRAIN_FEAT_HAND, index_col=0, usecols=['id', 'magic_max-freq','magic_min-freq', 'magic_intersect'])
    train_data = pd.concat([train_data, train_leak], axis=1)
    train_label = pd.read_csv(config.IN_TRAIN_LABEL, index_col=0)
    # build vocab
    wv_model = gensim.models.KeyedVectors.load(config.GLOBAL_GLOVE)
    vocab = build_vocab(train_data, wv_model)
    train_data[config.COLUMN_Q1] = train_data[config.COLUMN_Q1].apply(lambda x : vocab.numerize(x.split()))
    train_data[config.COLUMN_Q2] = train_data[config.COLUMN_Q2].apply(lambda x : vocab.numerize(x.split()))
    # train model
    train_data[config.COLUMN_LABEL] = train_label
    train_data, dev_data = train_test_split(train_data, test_size=0.01)
    fit(vocab, train_data, dev_data)


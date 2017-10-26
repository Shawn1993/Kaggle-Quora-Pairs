import os

# ------------------------------------------------------
# ------------------------ DIR -------------------------
# ------------------------------------------------------
ROOT_DIR = os.path.abspath('../..')
INPUT_DIR = '../Input'
OUTPUT_DIR = '../Output'
GLOBAL_DIR = '%s/Global' % ROOT_DIR
WORD2VEC_DIR = '%s/Word2Vec' % GLOBAL_DIR

# ------------------------------------------------------
# ------------------------ FILE ------------------------
# ------------------------------------------------------
TEST_DATA = '%s/test.csv' % INPUT_DIR
TRAIN_DATA = '%s/train.csv' % INPUT_DIR
TRAIN_LABEL = '%s/label.csv' % OUTPUT_DIR

GLOVE_MODEL = '%s/glove.6B.300d' % WORD2VEC_DIR
WORD2VEC_MODEL = '%s/GoogleNews-vectors-negative300' % WORD2VEC_DIR

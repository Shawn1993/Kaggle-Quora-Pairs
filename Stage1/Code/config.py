import os
# ------------------------------------------------------
# ------------------------ FILE ------------------------
# ------------------------------------------------------
GLOBAL_DIR = os.path.abspath('../../Global')
GLOBAL_KFOLD = '%s/kdf.pkl' % GLOBAL_DIR
GLOBAL_WORD2VEC_DIR = '%s/Word2Vec' % GLOBAL_DIR
GLOBAL_GLOVE = '%s/glove.6B.300d' % GLOBAL_WORD2VEC_DIR
GLOBAL_WORD2VEC = '%s/GoogleNews-vectors-negative300' % GLOBAL_WORD2VEC_DIR


ROOT_DIR = os.path.abspath('../')
INPUT_DIR = '%s/Input'% ROOT_DIR
IN_TRAIN_DIR = '%s/Train' % INPUT_DIR
IN_TEST_DIR = '%s/Test' % INPUT_DIR

IN_TRAIN_LABEL = '%s/label.csv' % INPUT_DIR

TMP_TRAIN_DIR = '%s/Tmp/Train' % ROOT_DIR
TMP_TEST_DIR = '%s/Tmp/Test' % ROOT_DIR


OUTPUT_DIR = '%s/Output'% ROOT_DIR
OUT_FEAT_IMPORTANCE = '%s/features_importance.csv' % OUTPUT_DIR
OUT_TRAIN_FEAT_DL = '%s/Train/features_dl.csv' % OUTPUT_DIR
OUT_TRAIN_FEAT_HAND = '%s/Train/features_hand.csv' % OUTPUT_DIR
OUT_TEST_FEAT_DL = '%s/Test/features_dl.csv' % OUTPUT_DIR
OUT_TEST_FEAT_HAND = '%s/Test/features_hand.csv' % OUTPUT_DIR

# ///////////////////// Stage 1 \\\\\\\\\\\\\\\\\\\\\\\\
RANDOM_STATE = 2017
# ------------------------------------------------------
# ------------------- Deep Learning --------------------
# ------------------------------------------------------
DL_MODEL = 'SelfAttentionBiLSTM'
DL_GPU = 1
# ------------------------------------------------------
# ------------------ Feature Columns -------------------
# ------------------------------------------------------
COLUMN_Q1 = 'question1'
COLUMN_Q2 = 'question2'
COLUMN_LABEL = 'is_duplicate'
# \\\\\\\\\\\\\\\\\\\\\\\\\--///////////////////////////

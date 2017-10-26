# -*- coding: utf-8 -*-


RANDOM_STATE = 2017

# ------------------------------------------------------
# ------------------------ FILE ------------------------
# ------------------------------------------------------
import os
ROOT_DIR = os.path.abspath('../')

INPUT_DIR = '%s/Input' % ROOT_DIR
IN_TRAIN_LABEL = '%s/Train/train_label.npy' % INPUT_DIR
IN_FEAT_IMPORTANCE = '%s/features_importance.csv' % INPUT_DIR
IN_TRAIN_FEAT = '%s/Train/features_160+9.npy' % INPUT_DIR
IN_TRAIN_FEAT_DL = '%s/Train/features_dl.csv' % INPUT_DIR
IN_TRAIN_FEAT_HAND = '%s/Train/features_hand.csv' % INPUT_DIR
IN_TEST_FEAT  = '%s/Test/features_160+9.npy' % INPUT_DIR
IN_TEST_FEAT_DL = '%s/Test/features_dl.csv' % INPUT_DIR
IN_TEST_FEAT_HAND = '%s/Test/features_hand.csv' % INPUT_DIR

OUTPUT_DIR = '%s/Output' % ROOT_DIR
OUTPUT_TRAIN_DIR = '%s/Train' % OUTPUT_DIR
OUTPUT_TEST_DIR = '%s/Test' % OUTPUT_DIR

GLOBAL_DIR = os.path.abspath('../../Global')
KFOLD_FILE = '%s/kdf.pkl' % GLOBAL_DIR

# ------------------------------------------------------
# ------------------ Feature Columns -------------------
# ------------------------------------------------------
COLUMN_Q1 = 'question1'
COLUMN_Q2 = 'question2'
COLUMN_LABEL = 'is_duplicate'



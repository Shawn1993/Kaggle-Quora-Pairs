import os
import nltk
import argparse
import pandas as pd
from datetime import datetime
from utils import dist_utils, ngram_utils

parser = argparse.ArgumentParser(description='Exctract the pos and ner features.')
parser.add_argument('-f', '--file', type=str, help='file name to process')
parser.add_argument('-p', '--prefix', type=str, help='prefix for features')
parser.add_argument('-d', '--save-dir', type=str, help='dir for save')
args = parser.parse_args()
if not args.file or not os.path.isfile(args.file):
    exit()
if not args.save_dir or not os.path.isdir(args.save_dir):
    exit()

NOUN = {'NN', 'NNS', 'NNP'}
VERB = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

def nltk_pos(words):
    return nltk.pos_tag(words)

def _noun_set(word_pos):
    return [word for word, pos in word_pos if pos in NOUN]

def _verb_set(word_pos):
    return [word for word, pos in word_pos if pos in VERB]

def noun_jaccard_ratio(a, b):
    a, b = _noun_set(a), _noun_set(b)
    return dist_utils.jaccard_ratio(a, b)

def verb_jaccard_ratio(a, b):
    a, b = _verb_set(a), _verb_set(b)
    return dist_utils.jaccard_ratio(a, b)

def noun_dice_ratio(a, b):
    a, b = _noun_set(a), _noun_set(b)
    return dist_utils.dice_ratio(a, b)

def verb_dice_ratio(a, b):
    a, b = _verb_set(a), _verb_set(b)
    return dist_utils.dice_ratio(a, b)


if __name__ == '__main__':
    start_time = datetime.now()
    df = pd.read_csv(args.file, index_col=0)
    print('Lines: %d' %len(df))
    # uniwords
    df['question1'] = df['question1'].apply(ngram_utils.uniwords)
    df['question2'] = df['question2'].apply(ngram_utils.uniwords)
    
    print('Use time: {}.'.format(datetime.now()-start_time))
    start_time = datetime.now()

    # pos
    df['question1'] = df['question1'].apply(nltk_pos)
    df['question2'] = df['question2'].apply(nltk_pos)
    
    print('Use time: {}.'.format(datetime.now()-start_time))
    start_time = datetime.now()
    
    # features
    sub_prefix = 'uniwords'
    df['%s_%s_noun-jaccard-ratio' % (args.prefix, sub_prefix)] = df.apply(lambda x: noun_jaccard_ratio(x['question1'], x['question2']), axis=1)
    df['%s_%s_verb-jaccard-ratio' % (args.prefix, sub_prefix)] = df.apply(lambda x: verb_jaccard_ratio(x['question1'], x['question2']), axis=1)
    df['%s_%s_noun-dice-ratio' % (args.prefix, sub_prefix)] = df.apply(lambda x: noun_dice_ratio(x['question1'], x['question2']), axis=1)
    df['%s_%s_verb-dice-ratio' % (args.prefix, sub_prefix)] = df.apply(lambda x: verb_dice_ratio(x['question1'], x['question2']), axis=1)
    print('Use time: {}.'.format(datetime.now()-start_time))
    start_time = datetime.now()

    # save
    df.drop(['question1', 'question2', 'is_duplicate'], axis=1, inplace=True, errors='ignore')
    save_path = os.path.join(args.save_dir, '%s_feature_parse.csv' % args.prefix)
    df.to_csv(save_path)
    print('Use time: {}. Save file to {}'.format(datetime.now()-start_time, save_path))








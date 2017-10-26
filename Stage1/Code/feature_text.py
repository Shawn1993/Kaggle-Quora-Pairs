import os
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


def run(df, ngram, prefix):
    # ngram
    df['q1_ngram'] = df['question1'].apply(ngram)
    df['q2_ngram'] = df['question2'].apply(ngram)

    ## 长度特征
    df['%s_q1-len' % prefix] = df['q1_ngram'].apply(len)
    df['%s_q2-len' % prefix] = df['q2_ngram'].apply(len)

    ## 字符集合相似度
    df['%s_dice-ratio'% prefix] = df.apply(lambda x: dist_utils.dice_ratio(x['q1_ngram'], x['q2_ngram']), axis=1)
    df['%s_jaccard-ratio'% prefix] = df.apply(lambda x: dist_utils.jaccard_ratio(x['q1_ngram'], x['q2_ngram']), axis=1)
    df['%s_edit-seq-ratio'% prefix] = df.apply(lambda x: dist_utils.edit_seq_ratio(x['q1_ngram'], x['q2_ngram']), axis=1)
    df['%s_edit-set-ratio'% prefix] = df.apply(lambda x: dist_utils.edit_set_ratio(x['q1_ngram'], x['q2_ngram']), axis=1)

    df.drop(['q1_ngram', 'q2_ngram'], axis=1, inplace=True)
    return df

if __name__ == '__main__':
    start_time = datetime.now()
    df = pd.read_csv(args.file, index_col=0, dtype=str)
    df = df.fillna('')
    print('Lines: %d' %len(df))

    df = run(df, ngram_utils.unichars, '%s_%s' % (args.prefix, 'unichars'))
    df = run(df, ngram_utils.bichars, '%s_%s' % (args.prefix, 'bichars'))
    df = run(df, ngram_utils.trichars, '%s_%s' % (args.prefix, 'trichars'))
    df = run(df, ngram_utils.uniwords, '%s_%s' % (args.prefix, 'uniwords'))
    df = run(df, ngram_utils.biwords, '%s_%s' % (args.prefix, 'biwords'))
    df = run(df, ngram_utils.triwords, '%s_%s' % (args.prefix, 'triwords'))

    # save
    df.drop(['question1', 'question2', 'is_duplicate'], axis=1, inplace=True, errors='ignore')
    save_path = os.path.join(args.save_dir, '%s_feature_text.csv' % args.prefix)
    df.to_csv(save_path)
    print('Use time: {}. Save file to {}'.format(datetime.now()-start_time, save_path))




        
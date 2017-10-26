import os
import argparse
import pandas as pd
from fuzzywuzzy import fuzz
from datetime import datetime

parser = argparse.ArgumentParser(description='Exctract the pos and ner features.')
parser.add_argument('-f', '--file', type=str, help='file name to process')
parser.add_argument('-p', '--prefix', type=str, help='prefix for features')
parser.add_argument('-d', '--save-dir', type=str, help='dir for save')
args = parser.parse_args()
if not args.file or not os.path.isfile(args.file):
    exit()
if not args.save_dir or not os.path.isdir(args.save_dir):
    exit()


def run(df, prefix):
    # 模糊匹配相似度
    df['%s_fuzz-qratio' % prefix] = df.apply(lambda x: fuzz.QRatio(x['question1'], x['question2']), axis=1)
    df['%s_fuzz-token-set-ratio' % prefix] = df.apply(lambda x: fuzz.token_set_ratio(x['question1'], x['question2']), axis=1)
    df['%s_fuzz_token_sort_ratio' % prefix] = df.apply(lambda x: fuzz.token_sort_ratio(x['question1'], x['question2']), axis=1)
    return df

if __name__ == '__main__':
    start_time = datetime.now()
    df = pd.read_csv(args.file, index_col=0)
    print('Lines: %d' %len(df))
    
    df = run(df, args.prefix)

    # save
    df.drop(['question1', 'question2', 'is_duplicate'], axis=1, inplace=True, errors='ignore')
    save_path = os.path.join(args.save_dir, '%s_feature_fuzz.csv' % args.prefix)
    df.to_csv(save_path)
    print('Use time: {}. Save file to {}'.format(datetime.now()-start_time, save_path))






import os
import pandas as pd
import argparse
from datetime import datetime
from collections import defaultdict

parser = argparse.ArgumentParser(description='Exctract the pos and ner features.')
parser.add_argument('--train-file', type=str, help='train file name to process')
parser.add_argument('--test-file', type=str, help='test file name to process')
parser.add_argument('--train-save-dir', type=str, help='train dir for save')
parser.add_argument('--test-save-dir', type=str, help='test dir for save')
args = parser.parse_args()
if not args.train_file or not os.path.isfile(args.train_file):
    print('error'), exit()
if not args.test_file or not os.path.isfile(args.test_file):
    print('error'), exit()
if not args.train_save_dir or not os.path.isdir(args.train_save_dir):
    print('error'), exit()
if not args.test_save_dir or not os.path.isdir(args.test_save_dir):
    print('error'), exit()

def magic1(train_data, test_data):
    all_data = pd.concat([
        train_data['question1'],
        train_data['question2'],
        test_data['question1'],
        test_data['question2']
    ], ignore_index=True)  
    all_data = all_data.apply(str)
    counter = all_data.value_counts().to_dict()

    train_data['magic_q1-freq'] = train_data['question1'].apply(counter.get)
    train_data['magic_q2-freq'] = train_data['question2'].apply(counter.get)
    test_data['magic_q1-freq'] = test_data['question1'].apply(counter.get)
    test_data['magic_q2-freq'] = test_data['question2'].apply(counter.get)
    return train_data, test_data

def magic2(train_data, test_data):
    ques = pd.concat([
        train_data[['question1', 'question2']],
        test_data[['question1', 'question2']]
    ], axis=0).reset_index(drop='index')

    q_dict = defaultdict(set)
    for i in range(len(ques)):
        q_dict[ques.question1[i]].add(ques.question2[i])
        q_dict[ques.question2[i]].add(ques.question1[i])

    def q1_q2_intersect(row):
        return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))

    train_data['magic_intersect'] = train_data.apply(q1_q2_intersect, axis=1)
    test_data['magic_intersect'] = test_data.apply(q1_q2_intersect, axis=1)
    return train_data, test_data

def run(train_data, test_data):
    return magic2(*magic1(train_data, test_data))

if __name__ == '__main__':
    start_time = datetime.now()
    train_data =  pd.read_csv(args.train_file, index_col=0, dtype=str)
    test_data =  pd.read_csv(args.test_file, index_col=0, dtype=str)
    
    train_data = train_data.fillna('')
    train_data = train_data.applymap(str)
    test_data = test_data.fillna('')
    test_data = test_data.applymap(str)

    train_data, test_data = run(train_data, test_data)
    train_data.drop(['question1','question2','qid1','qid2','is_duplicate'], axis=1, inplace=True, errors='ignore')
    test_data.drop(['question1','question2'], axis=1, inplace=True, errors='ignore')

    train_save_path = os.path.join(args.train_save_dir, 'feature_magic.csv')
    train_data.to_csv(train_save_path)
    print('Save train feature to %s' % train_save_path)
    test_save_path = os.path.join(args.test_save_dir, 'feature_magic.csv')
    test_data.to_csv(test_save_path)
    print('Save test feature to %s' % test_save_path)
    
    print('Use time: {}.'.format(datetime.now()-start_time))






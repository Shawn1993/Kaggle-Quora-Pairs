import os
import argparse
import pandas as pd
from datetime import datetime
from utils import dist_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

parser = argparse.ArgumentParser(description='Exctract the pos and ner features.')
parser.add_argument('-f', '--file', type=str, help='file name to process')
parser.add_argument('-p', '--prefix', type=str, help='prefix for features')
parser.add_argument('-d', '--save-dir', type=str, help='dir for save')
args = parser.parse_args()
if not args.file or not os.path.isfile(args.file):
    exit()
if not args.save_dir or not os.path.isdir(args.save_dir):
    exit()

class VectorSpace(object):

    def __init__(self, tfidf):
        self.tfidf = tfidf
        self.lsi = TruncatedSVD(n_components=100, random_state=2017)

    def transform(self, df1, df2):
        corpus = pd.concat([df1, df2], ignore_index=True)
        print('corpus')
        self.tfidf.fit(corpus)
        print('fitted tfidf')
        df1_lsi = self.lsi.fit_transform(self.tfidf.transform(df1))
        print('transform df1')
        df2_lsi = self.lsi.fit_transform(self.tfidf.transform(df2))
        print('transform df2')
        return list(map(dist_utils.cosine_distance, df1_lsi, df2_lsi))


class CharTfidfLSA(VectorSpace):
    """CharTfidfLSA"""

    def __init__(self):
        char_tfidf = TfidfVectorizer(min_df=3,
                                     max_df=0.75,
                                     norm="l2",
                                     strip_accents="unicode",
                                     analyzer="char",
                                     ngram_range=(1, 3),
                                     use_idf=1,
                                     smooth_idf=1,
                                     sublinear_tf=1)
        super(CharTfidfLSA, self).__init__(char_tfidf)


class WordTfidfLSA(VectorSpace):
    """CharTfidfLSA"""

    def __init__(self):
        word_tfidf = TfidfVectorizer(norm="l2",
                                     strip_accents="unicode",
                                     analyzer="word",
                                     ngram_range=(1, 3),
                                     use_idf=1,
                                     smooth_idf=1,
                                     sublinear_tf=1)
        super(WordTfidfLSA, self).__init__(word_tfidf)


def run(df, prefix):
    df['%s_chars-tfidf-lsa-cosine' %prefix] = CharTfidfLSA().transform(df['question1'], df['question2'])
    df['%s_words-tfidf-lsa-cosine' %prefix] = WordTfidfLSA().transform(df['question1'], df['question2'])
    return df

if __name__ == '__main__':
    start_time = datetime.now()
    df = pd.read_csv(args.file, index_col=0)
    print('Lines: %d' %len(df))
    
    df = run(df, args.prefix)

    # save
    df.drop(['question1', 'question2', 'is_duplicate'], axis=1, inplace=True, errors='ignore')
    save_path = os.path.join(args.save_dir, '%s_feature_lsi.csv' % args.prefix)
    df.to_csv(save_path)
    print('Use time: {}. Save file to {}'.format(datetime.now()-start_time, save_path))












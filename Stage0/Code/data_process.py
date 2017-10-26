import nltk
import regex as re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer
from string import punctuation
from collections import Counter
import gensim
import config
import multiprocessing
from itertools import chain
from datetime import datetime
import gc
from tqdm import tqdm
import pickle

class Transformer:

    def transform(self, text):
        raise NotImplementedError()


class LowerCaseConverter(Transformer):

    def transform(self, text):
        return str(text).lower()


class SpecialCharReplacer(Transformer):

    def transform(self, text):
        text = re.sub(r"？", "?", text)
        text = re.sub(r"，", ",", text)
        text = re.sub(r"、", ",", text)
        text = re.sub(r"。", ".", text)
        text = re.sub(r"：", ":", text)
        text = re.sub(r"；", ";", text)
        text = re.sub(r"！", "!", text)
        text = re.sub(r"～", "~", text)
        text = re.sub(r"—", "-", text)

        text = re.sub(r"“", '"', text)
        text = re.sub(r"”", '"', text)
        text = re.sub(r"【", "[", text)
        text = re.sub(r"】", "]", text)
        text = re.sub(r"《", " ", text)
        text = re.sub(r"》", " ", text)
        text = re.sub(r"「", " ", text)
        text = re.sub(r"」", " ", text)
        text = re.sub(r"（", "(", text)
        text = re.sub(r"）", ")", text)

        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"I'm ", "I am ", text)
        text = re.sub(r"'re ", " are ", text)
        text = re.sub(r"n't ", " not ", text)
        text = re.sub(r"'ve ", " have ", text)
        text = re.sub(r"'d ", " would ", text)
        text = re.sub(r"'ll ", " will ", text)
        text = re.sub(r"(\d+)k ", r"\1,000 ", text)
        text = re.sub(r"(\d)% ", r"\1 percent ", text)
        return text

# remove punctuation
class PunctuationRemover(Transformer):

    def __init__(self):
        self.pattern = r'[%s]' % punctuation

    def transform(self, text):
        return re.sub(self.pattern, ' ', text)

# tokenize
class Tokenizer(Transformer):

    def transform(self, text):
        return ' '.join(nltk.word_tokenize(text))


class WordSpellChecker:
    """https://www.kaggle.com/cpmpml/spell-checker-using-word2vec"""

    def __init__(self, model_name, corpus):
        self.model = gensim.models.KeyedVectors.load(model_name)
        self.WORDS = {w: i for i, w in enumerate(self.model.index2word)}
        tokens = [[w for w in s.split()] for s in corpus]
        freqs = Counter(chain(*tokens))
        self.dict = {w: self.correction(w) for w in tqdm(freqs)}

    def transform(self, text):
        return ' '.join([self.dict.get(w, w) for w in text.split()])

    def correction(self, word):
        '''Most probable spelling correction for word.'''
        return max(self.candidates(word), key=self.P)

    def P(self, word):
        '''Probability of `word`.'''
        # use inverse of rank as proxy
        # returns 0 if the word isn't in the dictionary
        return - self.WORDS.get(word, 0)

    def candidates(self, word):
        '''Generate possible spelling corrections for word.'''
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words):
        '''The subset of `words` that appear in the dictionary of WORDS.'''
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        '''All edits that are one edit away from `word`.'''
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        '''All edits that are two edits away from `word`.'''
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


# remove stopwords
class StopWordsRemover(Transformer):

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.stopwords.remove('can')

    def transform(self, text):
        return ' '.join([w for w in text.split() if w not in self.stopwords])


# lemmatize
class Lemmatizer(Transformer):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def transform(self, text):
        return ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])


# stem
class Stemmer(Transformer):

    def __init__(self):
        self.stemmer = SnowballStemmer('english')

    def transform(self, text):
        return ' '.join([self.stemmer.stem(word) for word in text.split()])


class ParallelProcessor:
    """
    WARNING: This class will operate on the original input dataframe itself

    https://stackoverflow.com/questions/26520781/multiprocessing-pool-whats-the-difference-between-map-async-and-imap
    """

    def __init__(self, processors, workers=1):
        self.processors = processors
        self.workers = workers

    def _process(self, df):
        print('B----------{}-----------'.format(multiprocessing.current_process().name))
        for processor in self.processors:
            start_time = datetime.now()
            print('Using [' + processor.__class__.__name__ + '] in ' + multiprocessing.current_process().name)
            df = df.apply(processor.transform)
            print('Finish [' + processor.__class__.__name__ + '] in ' + multiprocessing.current_process().name)
            print('Use Time: {}'.format(datetime.now() - start_time))
        print('E----------{}-----------'.format(multiprocessing.current_process().name))
        print()
        return df

    def process(self, df, columns):
        if self.workers < 2:
            results = map(self._process, [df[col] for col in columns])
        else:
            pool = multiprocessing.Pool(self.workers)
            results = pool.imap(self._process, [df[col] for col in columns])
            pool.terminate()

        for col, res in zip(columns, results):
            df[col] = res
        return df


def process(processor, df, save_fname):
    print('Processing...\n')
    process_columns = ['question1', 'question2']
    df = processor.process(df.copy(), process_columns)
    df.to_csv(save_fname)
    print('Saved file to "%s"\n\n\n\n' % save_fname)
    return df

if __name__ == '__main__':

    start_time = datetime.now()
    # Train
    train = pd.read_csv(config.TRAIN_DATA, index_col=0)
    train[['is_duplicate']].to_csv(config.TRAIN_LABEL)
    train = train.drop(['q1id', 'q2id', 'is_duplicate'], axis=1, errors='ignore')
    test = pd.read_csv(config.TEST_DATA, index_col=0)

    # tokenize
    p_tokenize = ParallelProcessor(
        [
            LowerCaseConverter(),  # must
            SpecialCharReplacer(),  # must
            PunctuationRemover(),  # must
            Tokenizer(),  # must
        ]
    )
    train_tokenize = process(p_tokenize, train, config.TRAIN_DATA_TOKENIZE)
    test_tokenize = process(p_tokenize, test, config.TEST_DATA_TOKENIZE)

    # stopword
    p_stopword = ParallelProcessor(
        [
            StopWordsRemover()  # option
        ]
    )
    train_stopword = process(p_stopword, train_tokenize, config.TRAIN_DATA_STOPWORD)
    test_stopword = process(p_stopword, test_tokenize, config.TEST_DATA_STOPWORD)
    del train_tokenize, test_tokenize
    gc.collect()

    # letimized
    p_lemmatize = ParallelProcessor(
        [
            Lemmatizer()
        ]
    )
    train_lemmatized = process(p_lemmatize, train_stopword, config.TRAIN_DATA_LEMMATIZED)
    process(p_stem, train_lemmatized, config.TRAIN_DATA_LEMMATIZED_STEMMED)
    gc.collect()

    test_lemmatized = process(p_lemmatize, test_stopword, config.TEST_DATA_LEMMATIZED)
    process(p_stem, test_lemmatized, config.TEST_DATA_LEMMATIZED_STEMMED)
    gc.collect()

    # make corpus
    corpus = pd.concat([
        train_lemmatized['question1'],
        train_lemmatized['question2'],
        test_lemmatized['question1'],
        test_lemmatized['question2']
    ])

    # # word2vec spell check
    # word2vec_spellchecker = WordSpellChecker(model_name=config.WORD2VEC_MODEL, corpus=corpus)
    # pickle.dump(word2vec_spellchecker.dict, open(config.WORD2VEC_SPELL_CHECHER_DATA,'wb'), protocol=2)
    # p_check_word2vec = ParallelProcessor(
    #     [
    #         word2vec_spellchecker
    #     ]
    # )
    
    # # glove spell check
    # glove_spellchecker = WordSpellChecker(model_name=config.GLOVE_MODEL, corpus=corpus)
    # pickle.dump(glove_spellchecker.dict, open(config.GLOVE_SPELL_CHECHER_DATA,'wb'), protocol=2)
    # p_check_glove = ParallelProcessor(
    #     [
    #         glove_spellchecker 
    #     ]
    # )

    # Stem
    p_stem = ParallelProcessor(
        [
            Stemmer()  # option
        ]
    )

    train_lemmatized_glove = process(p_check_glove, train_lemmatized, config.TRAIN_DATA_LEMMATIZED_GLOVE)
    process(p_stem, train_lemmatized_glove, config.TRAIN_DATA_LEMMATIZED_GLOVE_STEMMED)
    del train_lemmatized_glove
    gc.collect()

    train_lemmatized_word2vec = process(p_check_word2vec, train_lemmatized, config.TRAIN_DATA_LEMMATIZED_WORD2VEC)
    process(p_stem, train_lemmatized_word2vec, config.TRAIN_DATA_LEMMATIZED_WORD2VEC_STEMMED)
    del train_lemmatized_word2vec
    gc.collect()

    test_lemmatized_glove = process(p_check_glove, test_lemmatized, config.TEST_DATA_LEMMATIZED_GLOVE)
    process(p_stem, test_lemmatized_glove, config.TEST_DATA_LEMMATIZED_GLOVE_STEMMED)
    del test_lemmatized_glove
    gc.collect()

    test_lemmatized_word2vec = process(p_check_word2vec, test_lemmatized, config.TEST_DATA_LEMMATIZED_WORD2VEC)
    process(p_stem, test_lemmatized_word2vec, config.TEST_DATA_LEMMATIZED_WORD2VEC_STEMMED)
    del test_lemmatized_word2vec
    gc.collect()

    print('All Time Used: {}'.format(datetime.now()-start_time))


    

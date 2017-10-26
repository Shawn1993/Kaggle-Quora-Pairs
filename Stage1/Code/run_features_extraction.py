import os
import config

# args_to_process = [
#     # train
#     (config.TRAIN_DATA_LEMMATIZED, 'lemmatized', config.FEAT_TRAIN_DIR),
#     (config.TRAIN_DATA_LEMMATIZED_GLOVE, 'lemmatized_glove', config.FEAT_TRAIN_DIR),
#     (config.TRAIN_DATA_LEMMATIZED_WORD2VEC, 'lemmatized_word2vec', config.FEAT_TRAIN_DIR),
#     (config.TRAIN_DATA_LEMMATIZED_STEMMED, 'lemmatized_stemmed', config.FEAT_TRAIN_DIR),
#     (config.TRAIN_DATA_LEMMATIZED_GLOVE_STEMMED,'lemmatized_glove_stemmed', config.FEAT_TRAIN_DIR),
#     (config.TRAIN_DATA_LEMMATIZED_WORD2VEC_STEMMED,'lemmatized_word2vec_stemmed', config.FEAT_TRAIN_DIR),
#     # test
#     (config.TEST_DATA_LEMMATIZED, 'lemmatized', config.FEAT_TEST_DIR),
#     (config.TEST_DATA_LEMMATIZED_GLOVE, 'lemmatized_glove', config.FEAT_TEST_DIR),
#     (config.TEST_DATA_LEMMATIZED_WORD2VEC, 'lemmatized_word2vec', config.FEAT_TEST_DIR),
#     (config.TEST_DATA_LEMMATIZED_STEMMED, 'lemmatized_stemmed', config.FEAT_TEST_DIR),
#     (config.TEST_DATA_LEMMATIZED_GLOVE_STEMMED,'lemmatized_glove_stemmed', config.FEAT_TEST_DIR),
#     (config.TEST_DATA_LEMMATIZED_WORD2VEC_STEMMED,'lemmatized_word2vec_stemmed', config.FEAT_TEST_DIR)      
# ]

args_to_dl = [
    # (
    #     config.TRAIN_DATA_LCS,
    #     config.TEST_DATA_LCS,
    #     'lcs',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ),
    # (
    #     config.TRAIN_DATA_TOKENIZE,
    #     config.TEST_DATA_TOKENIZE,
    #     'tokenize',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ),
    # (
    #     config.TRAIN_DATA_STOPWORD,
    #     config.TEST_DATA_STOPWORD,
    #     'tokenize_stopword',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ),
    # (
    #     config.TRAIN_DATA_STOPWORD_ANV,
    #     config.TEST_DATA_STOPWORD_ANV,
    #     'tokenize_stopword_anv',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ),
    # (
    #     config.TRAIN_DATA_LEMMATIZED,
    #     config.TEST_DATA_LEMMATIZED,
    #     'lemmatized',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ), (
    #     config.TRAIN_DATA_LEMMATIZED_GLOVE,
    #     config.TEST_DATA_LEMMATIZED_GLOVE,
    #     'lemmatized_glove',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ), 
    # (
    #     config.TRAIN_DATA_LEMMATIZED_WORD2VEC,
    #     config.TEST_DATA_LEMMATIZED_WORD2VEC,
    #     'lemmatized_word2vec',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.WORD2VEC_MODEL
    # ), (
    #     config.TRAIN_DATA_LEMMATIZED_STEMMED,
    #     config.TEST_DATA_LEMMATIZED_STEMMED,
    #     'lemmatized_stemmed',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ), 
    # (
    #     config.TRAIN_DATA_LEMMATIZED_GLOVE_STEMMED,
    #     config.TEST_DATA_LEMMATIZED_GLOVE_STEMMED,
    #     'lemmatized_glove_stemmed',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.GLOVE_MODEL
    # ), (
    #     config.TRAIN_DATA_LEMMATIZED_WORD2VEC_STEMMED,
    #     config.TEST_DATA_LEMMATIZED_WORD2VEC_STEMMED,
    #     'lemmatized_word2vec_stemmed',
    #     config.FEAT_TRAIN_DIR,
    #     config.FEAT_TEST_DIR,
    #     'SelfAttentionBiLSTM',
    #     config.WORD2VEC_MODEL
    # ),

]


python = 'python3'

if __name__ == '__main__':
    # for args in args_to_process:
    #     # feature_parse
    #     cmd = '%s feature_parse.py -f=%s -p=%s -d=%s' % (python, *args)
    #     os.system(cmd)

    #     # feature_text
    #     cmd = '%s feature_text.py -f=%s -p=%s -d=%s' % (python, *args)
    #     os.system(cmd)

    #     # feature_embed
    #     cmd = '%s feature_embed.py -f=%s -p=%s -d=%s' % (python, *args)
    #     print('[Run command]:\n' + cmd)
    #     os.system(cmd)
        
    #     # feature_fuzz
    #     cmd = '%s feature_fuzz.py -f=%s -p=%s -d=%s' % (python, *args)
    #     os.system(cmd)

    #     # feature_lsi
    #     cmd = '%s feature_lsi.py -f=%s -p=%s -d=%s' % (python, *args)
    #     os.system(cmd)
    #     pass

    # dl features
    for args in args_to_dl:
        cmd = '''%s feature_dl.py \\
                --train-file=%s \\
                --test-file=%s \\
                -p=%s \\
                --train-save-dir=%s \\
                --test-save-dir=%s \\
                --dl-model=%s \\
                --wv-model=%s''' %(python, *args)
        print('[Run command]:\n' + cmd)
        os.system(cmd)

    # magic features
    # args = (config.TRAIN_DATA, config.TEST_DATA, config.FEAT_TRAIN_DIR, config.FEAT_TEST_DIR)
    # cmd = '''%s feature_magic.py \\
    #             --train-file=%s \\
    #             --test-file=%s \\
    #             --train-save-dir=%s \\
    #             --test-save-dir=%s ''' %(python, *args) 
    # print('[Run command]:\n' + cmd)
    # os.system(cmd)

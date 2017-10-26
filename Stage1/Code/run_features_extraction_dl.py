import os
import config
import glob
from os.path import basename, splitext, join


def ask_files():
    files = glob.glob(config.IN_TRAIN_DIR+'/*')
    names = [splitext(basename(x))[0] for x in files]
    nums = [str(i) for i in range(len(names))]
    input_num = input('Which File to be used?   Note: use "," split for multi-choice.\n%s\n-> ' %('\n'.join(map(lambda x,y:x+')\t'+y, nums, names)))).strip()
    while True:
        try:
            for num in input_num.split(','):
                assert num.strip() in nums
            else:
                break
        except :
            pass
        input_num = input('Please input the true number in true format:\n-> ').strip()
    print('You choose :\n%s\n'%('\n'.join([names[int(x.strip())] for x in input_num.split(',')])))
    return [basename(files[int(x.strip())]) for x in input_num.split(',')]

def ask_word2vec():
    files = glob.glob(config.GLOBAL_WORD2VEC_DIR+'/*.syn0.npy')
    names = [basename(x).replace('.syn0.npy','') for x in files]
    nums = [str(i) for i in range(len(names))]
    input_num = input('Which Word2Vec model to be used?   Note: use "," split for multi-choice.\n%s\n-> ' %('\n'.join(map(lambda x,y:x+')\t'+y, nums, names)))).strip()
    while True:
        try:
            for num in input_num.split(','):
                assert num.strip() in nums
            else:
                break
        except :
            pass
        input_num = input('Please input the true number in true format:\n-> ').strip()
    print('You choose :\n%s\n'%('\n'.join([names[int(x.strip())] for x in input_num.split(',')])))
    return [names[int(x.strip())] for x in input_num.split(',')]



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Use to feature selcection')
    parser.add_argument('-auto', action="store_true", help='use which method')
    args = parser.parse_args()

    if args.auto :
        input_files = [basename(x) for x in glob.glob(config.IN_TRAIN_DIR+'/*')]
        w2v_files = [basename(x).replace('.syn0.npy','') for x in glob.glob(config.GLOBAL_WORD2VEC_DIR+'/*.syn0.npy')]
    else :
        input_files = ask_files()
        w2v_files = ask_word2vec()


    python = 'python3'
    for i in input_files:
        for w in w2v_files:
            cmd = '''%s feature_dl_leak.py \\
                    --train-file=%s \\
                    --test-file=%s \\
                    -p=%s \\
                    --gpu=%d \\
                    --train-save-dir=%s \\
                    --test-save-dir=%s \\
                    --dl-model=%s \\
                    --wv-model=%s''' % (
                python,
                join(config.IN_TRAIN_DIR, i),
                join(config.IN_TEST_DIR, i),
                splitext(i)[0] + '_' + w,
                1,
                config.TMP_TRAIN_DIR,
                config.TMP_TEST_DIR,
                'SimpleCNN',
                join(config.GLOBAL_WORD2VEC_DIR, w)
            )
            print('[Run command]:\n' + cmd)
            os.system(cmd)

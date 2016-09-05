import argparse
from argparse import ArgumentParser
import os
import write_normalized_txt
import write_word2vec

p = ArgumentParser(description='Harry Potter Machine - Generates Text from Given Books')
p.add_argument('-b', '--book', default='harry-potter')
p.add_argument('-s', '--sourcefile')
p.add_argument('-v', '--vocab', type=argparse.FileType('r'))
p.add_argument('-n', '--nocache', action='store_true', default=False)
p.add_argument('--sampleonly', action='store_true', default=False)

args = p.parse_args()


def create_dir_if_not_exists(dirname):
    try:
        os.stat(dirname)
    except:
        print("creating {0} dir".format(dirname))
        os.makedirs(dirname)


if __name__ == '__main__':
    print(args)

    if not args.sampleonly:

        print('running training pipeline... ')

        create_dir_if_not_exists('data/' + args.book)

        # Find / create the cleaned data
        os.chdir('data/' + args.book)
        if args.nocache or not (os.path.exists('raw.txt')):
            print('normalizing input txt ...')
            write_normalized_txt.writefile(args.sourcefile, 'raw.txt')
            print('... wrote txt')

        # Find / create the vocab
        if args.vocab is None and (args.nocache or not os.path.exists('vocab.txt')):
            print('training vocab ... ')
            write_word2vec.train('raw.txt', 'vocab.txt')
            print('... vocab trained')

        # Create vectorized vocab iterator

        # (Continue) Training the model

        # Sample the model

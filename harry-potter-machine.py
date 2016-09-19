import argparse
from argparse import ArgumentParser
import os
import write_normalized_txt
import write_word2vec
import word2vec_api
import train_lstm
import reader

p = ArgumentParser(description='Harry Potter Machine - Generates Text from Given Books')
p.add_argument('-b', '--book', default='harry-potter')
p.add_argument('-s', '--sourcefile')
p.add_argument('-v', '--vocab', type=argparse.FileType('r'))
p.add_argument('-n', '--nocache', action='store_true', default=False)
p.add_argument('--sampleonly', action='store_true', default=False)
p.add_argument('--word2vec_complexity', type=int, default=300)

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

        create_dir_if_not_exists(os.path.join('data', args.book))

        # Find / create the cleaned data
        os.chdir(os.path.join('data', args.book))
        if args.nocache or not (os.path.exists('raw.txt')):
            print('normalizing input txt ...')
            write_normalized_txt.writefile(args.sourcefile, 'raw.txt')
            print('... wrote txt')

        has_vocab = os.path.exists('vocab.txt')
        wrong_size = False
        if has_vocab: # Retrain vector model if its different dimensions
            with open('vocab.txt') as f:
                size = int(f.readline().split(' ')[1])
                print('existing vocab size ' + str(size))
                wrong_size = size != args.word2vec_complexity

        # Find / create the vocab
        if args.vocab is None and (args.nocache or not has_vocab or wrong_size):
            print('training vocab ... ')
            write_word2vec.train(inputfile='raw.txt', outputfile='vocab.txt', complexity=args.word2vec_complexity)
            print('... vocab trained')

        data_sets = {
            'train.txt': 0.4,
            'test.txt': 0.3,
            'validate.txt': 0.3
        }
        missing_some = len([filename for filename in data_sets.keys() if os.path.isfile(filename)]) < len(data_sets)
        if args.nocache or missing_some:
            print('splitting data sets ...')
            reader.split('raw.txt', data_sets)
            print('... data sets split')

    word2vec_model = word2vec_api.from_model_at(args.vocab if args.vocab is not None else 'vocab.txt')

    if not args.sampleonly:
        print(word2vec_model.vector_to_word(word2vec_model.word_to_vector('The')))
        print('starting training ... ')
        train_lstm.train(word2vec_model, reader.Text('train.txt', word2vec_model))
        # train
        print('... training done? ')



    # (Continue) Training the model

    # Sample the model

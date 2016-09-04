from gensim.models.word2vec import Word2Vec
from nltk import tokenize


def file_to_sentence_tokens(filename: str):
    two_d_sents = (tokenize.sent_tokenize(line) for line in open(filename))
    return (tokenize.word_tokenize(sent) for sents in two_d_sents for sent in sents)


def run():
    fname = 'harry-potter.txt'

    def generator():
        return file_to_sentence_tokens('train/' + fname)

    model = Word2Vec(min_count=1, workers=16)
    model.build_vocab(sentences=generator())
    model.train(sentences=generator())
    model.save_word2vec_format('save/' + fname)


if __name__ == '__main__':
    run()





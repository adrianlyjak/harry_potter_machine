from nltk import tokenize
from gensim.models.word2vec import Word2Vec
import multiprocessing


def file_to_sentence_tokens(filename: str):
    two_d_sents = (tokenize.sent_tokenize(line) for line in open(filename))
    return (tokenize.word_tokenize(sent) for sents in two_d_sents for sent in sents)


def train(inputfile: str, outputfile: str, complexity: int):
    def generator():
        return file_to_sentence_tokens(inputfile)

    model = Word2Vec(min_count=1, workers=multiprocessing.cpu_count(), size=complexity)
    model.build_vocab(sentences=generator())
    model.train(sentences=generator())
    model.save_word2vec_format(outputfile)

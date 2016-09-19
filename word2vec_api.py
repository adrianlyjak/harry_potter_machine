from gensim.models.word2vec import Word2Vec
import numpy as np
from flask import Flask, request, jsonify
from nltk import tokenize as toke


class W2VException(Exception):
    def __init__(self, code, message):
        super(W2VException, self).__init__(self, message)
        self.code = code


class W2V(object):
    def __init__(self, model: Word2Vec):
        self.model = model

    def words_to_vectors(self, words):
        tokens = toke.word_tokenize(words)
        vecs = [self.model[t].tolist() for t in tokens if t in self.model]
        return vecs

    def word_to_vector(self, word) -> np.array:
        if word in self.model:
            return self.model[word].tolist()
        else:
            raise W2VException('unknown_word', 'unknown word: "' + word + '"')

    def vector_to_word(self, vector):
        words = self.model.similar_by_vector(np.array(vector), topn=1)
        return words[0][0]

    def vectors_to_words(self, vectors):
        words = [self.model.similar_by_vector(np.array(vec), topn=1)[0][0] for vec in vectors]
        return words

    # TODO write this one
    def vectors_to_formatted_text(self):
        return ''

    def width(self):
        return len(self.word_to_vector('the'))



def create_api(interface: W2V):
    api = Flask(__name__)

    docs = '''
    /toVector/many
        POST: {data: String} -> {result: Number[][]}
    /toVector/one/<word>
        GET: {data: Number[]}
    /toWord/one
        POST: {data: Number[]} -> {result: String}
    /toWord/many
        POST: {data: Number[][]} -> {result: String[]}

    /toWord/many/text
        POST: {data: Number[][]} -> {result: String}
    '''

    def err(code: str, reason: str):
        return jsonify(code=code, reason=reason)

    @api.route('/')
    def docs():
        return docs

    @api.route('/toVector/many', methods=['POST'])
    def words_to_vectors():
        words = request.get_json()['data']
        return jsonify(result=interface.words_to_vectors(words))

    @api.route('/toVector/one/<word>')
    def word_to_vector(word):
        try:
            return jsonify(result=interface.word_to_vector(word))
        except W2VException as e:
            return err(e.code, str(e))

    @api.route('/toWord/one', methods=['POST'])
    def vector_to_word():
        word = interface.vector_to_word(np.array(request.get_json()['data']))
        return jsonify(result=word)

    @api.route('/toWord/many', methods=['POST'])
    def vectors_to_words():
        words = interface.vectors_to_words(request.get_json()['data'])
        return jsonify(result=words)

    # TODO write this one
    @api.route('/toWord/many/text', methods=['POST'])
    def vectors_to_formatted_text():
        vectors = request.get_json()['data']
        return jsonify(result='')

    @api.route('/toVector/width')
    def width():
        return jsonify(result=interface.width())

    return api


def flask_from_model_at(filename: str):
    return create_api(from_model_at(filename))


def from_model_at(filename: str):
    return W2V(Word2Vec.load_word2vec_format(filename))

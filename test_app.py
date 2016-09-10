import unittest as ut

import word2vec_api
from flask import json
from numpy.testing import assert_almost_equal


class AppTest(ut.TestCase):

    def setUp(self):
        self.app = word2vec_api.flask_from_model_at('test-models/word.txt')
        self.app.testing = True
        self.client = self.app.test_client()

    def test_words_to_vectors(self):
        response = self.postJson('/toVector/many', {
            'data': 'the word'
        })
        expected = [[0.1, 0.9], [0.2, 0.8]]
        assert_almost_equal(response['result'], expected, 7)

    def test_words_to_vectors_word_missing(self):
        response = self.postJson('/toVector/many', {
            'data': 'no word'
        })
        assert_almost_equal(response['result'], [[0.2, 0.8]])

    def test_word_to_vec(self):
        response = json.loads(self.client.get('/toVector/one/word').data)
        assert_almost_equal(response['result'], [0.2, 0.8])

    def test_vec_to_word(self):
        response = self.postJson('/toWord/one', {
            'data': [0.2, 0.75]
        })
        self.assertEqual('word', response['result'])

    def test_vecs_to_words(self):
        response = self.postJson('/toWord/many', {
            'data': [[0.2, 0.75]]
        })
        self.assertEqual(['word'], response['result'])

    def testWidth(self):
        result = json.loads(self.client.get('/toVector/width').data)['result']
        self.assertEqual(result, 2)

    def postJson(self, path: str, data: dict) -> dict:
        return json.loads(self.client.post(path, data=json.dumps(data), content_type='application/json').data)




if __name__ == '__main__':
    ut.main()

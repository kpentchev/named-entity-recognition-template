import tensorflow as tf
from nltk import download, sent_tokenize
from NerModel import restore

MAX_LEN = 75
MAX_LEN_CHARS = 15

class Predictor(object):
    """Predictor service using an LSTM model"""
    def __init__(self):
        download('punkt')
        download('averaged_perceptron_tagger')
        download('stopwords')

        self.model = restore('/Users/kpentchev/data/models/2019_03_22_16_38_stem_char_lstm_crf.h5')
        self.graph = tf.get_default_graph()

    def predict(self, text):
        sentences = sent_tokenize(text)

        result = []

        for sentence in sentences:
            with self.graph.as_default():
                prediction = self.model.predict(sentence)
                for w, pred1 in prediction:
                    result.append((w[0], pred1))
        
        return result

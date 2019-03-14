import tensorflow as tf
from nltk import download, sent_tokenize, word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer
from WordIndex import WordIndex
from Preprocessor import encodeSentences, encodeStems, encodeChars, pad
from StemCharEmbLstmCrfModel import restore

MAX_LEN = 75
MAX_LEN_CHARS = 15

class Predictor(object):
    """Predictor service using an LSTM model"""
    def __init__(self):
        download('punkt')
        download('averaged_perceptron_tagger')
        download('stopwords')

        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)

        self.wordIndex = WordIndex("UNK")
        self.wordIndex.load('models_active/word_to_index.pickle')

        self.tagIndex = WordIndex("O")
        self.tagIndex.load('models_active/tag_to_index.pickle')

        self.charIndex = WordIndex("UNK")
        self.charIndex.load('models_active/char_to_index.pickle')

        self.stemIndex = WordIndex("UNK")
        self.stemIndex.load('models_active/stem_to_index.pickle')

        self.model = restore('models_active/stem_char_lstm_crf.h5')
        self.graph = tf.get_default_graph()

    def predict(self, text):
        sentences = sent_tokenize(text)

        result = []

        for sentence in sentences:
            words = pos_tag(word_tokenize(sentence))
            encodedInput = encodeSentences([words], self.wordIndex)
            encodedInput = pad(encodedInput, MAX_LEN, self.wordIndex.getPadIdx())

            encodedStems = encodeStems([[w[0] for w in words]], self.stemIndex, self.stemmer)
            encodedStems = pad(encodedStems, MAX_LEN, self.stemIndex.getPadIdx())

            encodedChars = encodeChars([words], self.charIndex, MAX_LEN, MAX_LEN_CHARS)

            with self.graph.as_default():
                prediction = self.model.predict(encodedInput, encodedStems, encodedChars)
                for w, pred1 in zip(words, prediction[0]):
                    result.append((w[0], self.tagIndex.getWord(pred1)))
        
        return result

from NerModel import NerModel
from WordIndex import WordIndex
from SentenceGetter import SentenceGetter
from Preprocessor import encodeSentences, encodeStems, encodeChars, encodeTags, pad, onehotEncodeTags
from ElmoEmbedding import ElmoEmbedding

import keras.backend as K
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D, Conv1D, Flatten, MaxPooling1D, Lambda
from keras.layers.merge import add
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tqdm import tqdm

class ElmoLstmCrfModel(NerModel):
    """Class representing an Char Embedding + LSTM CRF Model for NER"""
    def __init__(self, maxLengthSentence=0):
        super().__init__()

        self.maxLengthSentence = maxLengthSentence

        #lazy init model after preprocessing training data
        #word sizes 10% +, chars, tags exact

    def train(self, data, batch_size, n_epochs, test_size=0.1):
        x_words, y_enc = self.__preprocess(data)
        if self.model == None:
            self.__init_model()
        return self.__train(x_words, y_enc, batch_size, n_epochs, test_size)

    def __init_model(self):
        if (True):
            # Model definition
            inputWords = Input(shape=(self.maxLengthSentence,), dtype='string')
            embeddingWords = ElmoEmbedding()(inputWords)

            #x = Bidirectional(LSTM(units=512, return_sequences=True,
            #           recurrent_dropout=0.2, dropout=0.2))(embeddingWords)
            x_rnn = Bidirectional(LSTM(units=70, return_sequences=True,
                                    recurrent_dropout=0.2, dropout=0.2))(embeddingWords)

            #x_add = add([x, x_rnn])  # residual connection to the first biLSTM
            nn = TimeDistributed(Dense(70, activation="relu"))(x_rnn)  # a dense layer as suggested by neuralNer
            crf = CRF(self.nTags+1)  # CRF layer, n_tags+1(PAD)
            out = crf(nn)  # output

            self.model = Model([inputWords], out)
            self.model.compile(optimizer="nadam", loss=crf_loss, metrics=[crf_viterbi_accuracy]) #try adam optimizer
            self.model.summary()

    def __preprocess(self, data):
        print("Number of sentences: ", len(data.groupby(['Sentence #'])))

        tags = list(set(data["Tag"].values))
        print("Tags:", tags)
        self.nTags = len(tags)
        print("Number of Tags: ", self.nTags)

        if self.model == None:
            self.tagIndex = WordIndex("O", self.nTags+2)

        self.tagIndex.add(tags)

        getter = SentenceGetter(data)

        sentences = getter.sentences

        paddedSentences = padSentences(sentences, self.maxLengthSentence)

        encodedTags = encodeTags(sentences, self.tagIndex)
        encodedTags = pad(encodedTags, self.maxLengthSentence, self.tagIndex.getPadIdx())
        encodedTags = onehotEncodeTags(encodedTags, self.nTags)

        return  (paddedSentences, encodedTags)

    def __train(self, xWords, y, batchSize, epochs, testSize):
        X_word_tr, X_word_te, y_tr, y_te = train_test_split(xWords, y, test_size=testSize, random_state=2018)

        self.X_word_tr = np.array(X_word_tr)
        self.X_word_te = np.array(X_word_te)
        self.y_tr = np.array(y_tr)
        self.y_te = np.array(y_te)

        early_stopping = EarlyStopping(monitor='loss', min_delta=0.0050, patience=2, verbose=1)

        self.history = self.model.fit(
                                    self.X_word_tr,
                                    self.y_tr,
                                    batch_size=batchSize, 
                                    epochs=epochs,
                                    validation_data=(self.X_word_te, self.y_te),
                                    callbacks=[early_stopping],
                                    verbose=2)

    def evaluate(self):
        if len(self.X_word_te[0]) == 0:
            return

        pred_cat = self.model.predict(self.X_word_te)
        pred = np.argmax(pred_cat, axis=-1)
        y_te_true = np.argmax(self.y_te, -1)

        # Convert the index to tag
        pred_tag = [[self.tagIndex.getWord(i) for i in row] for row in pred]
        y_te_true_tag = [[self.tagIndex.getWord(i) for i in row] for row in y_te_true] 

        report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
        print(report)

    def predict(self, sentence):
        words = nltk.pos_tag(nltk.word_tokenize(sentence))
        
        encodedInput = np.array(padSentences([words], self.maxLengthSentence))

        prediction = self.model.predict(encodedInput)
        prediction = np.argmax(prediction, axis=-1)
        return zip(words, [self.tagIndex.getWord(p) for p in prediction[0]])

def padSentences(x, max_len):
    new_x = []
    for seq in x:
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i][0])
            except:
                new_seq.append("--PAD--")
        new_x.append(new_seq)
    return new_x

    

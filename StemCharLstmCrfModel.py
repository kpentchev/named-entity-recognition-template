from NerModel import NerModel
from WordIndex import WordIndex
from SentenceGetter import SentenceGetter
from Preprocessor import encodeSentences, encodeStems, encodeChars, encodeTags, pad, onehotEncodeTags

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D, Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report

import numpy as np

from tqdm import tqdm

class StemCharLstmCrfModel(NerModel):
    """Class representing an Char Embedding + LSTM CRF Model for NER"""
    def __init__(self, nWords=0, nChars=0, nTags=0, lenEmdSent=0, lenEmdWord=0, maxLengthSentence=0, maxLengthWord=0):
        super().__init__()

        nltk.download('stopwords')
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.wordIndex = WordIndex("UNK")

        self.stemIndex = WordIndex("UNK")

        self.tagIndex = WordIndex("O")

        self.charIndex = WordIndex("UNK")

        if (nWords > 0 and nTags > 0 and nChars > 0):
            self.maxLengthSentence = maxLengthSentence
            self.maxLengthWord = maxLengthWord
            self.nWords = nWords
            self.nChars = nChars
            self.nTags = nTags
            

            # Model definition
            inputWords = Input(shape=(maxLengthSentence,))
            embeddingWords = Embedding(input_dim=nWords+2, output_dim=lenEmdSent, # n_words + 2 (PAD & UNK)
                            input_length=maxLengthSentence, mask_zero=True)(inputWords)  # default: 20-dim embedding

            inputStems = Input(shape=(maxLengthSentence,))
            embeddingStems = Embedding(input_dim=nWords+2, output_dim=lenEmdSent, # n_words + 2 (PAD & UNK)
                            input_length=maxLengthSentence, mask_zero=True)(inputStems)  # default: 20-dim embedding

            inputChars = Input(shape=(maxLengthSentence, maxLengthWord,))
            embeddingChars = TimeDistributed(Embedding(input_dim=nChars + 2, output_dim=lenEmdWord, # n_chars + 2 (PAD & UNK)
                            input_length=maxLengthWord, mask_zero=False))(inputChars)

            charDropout = Dropout(0.5)(embeddingChars)
            charCnn = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(charDropout) #convolution over characters; faster than lstm
            maxpoolOut = TimeDistributed(MaxPooling1D(maxLengthWord), name="Maxpool")(charCnn)
            charCnnOut = TimeDistributed(Flatten(), name="Flatten")(maxpoolOut)
            charCnnOut = Dropout(0.6)(charCnnOut)
            
            embeddingCombined = SpatialDropout1D(0.4)(concatenate([embeddingWords, embeddingStems, charCnnOut]))

            mainLstmOne = Bidirectional(LSTM(units=70, return_sequences=True,
                                    recurrent_dropout=0.5))(embeddingCombined)  # variational biLSTM
            mainLstmTwo = Bidirectional(LSTM(units=70, return_sequences=True,
                                    recurrent_dropout=0.25))(mainLstmOne)

            nn = TimeDistributed(Dense(70, activation="relu"))(mainLstmTwo)  # a dense layer as suggested by neuralNer
            crf = CRF(nTags+1)  # CRF layer, n_tags+1(PAD)
            out = crf(nn)  # output

            self.model = Model([inputWords, inputStems, inputChars], out)
            self.model.compile(optimizer="nadam", loss=crf_loss, metrics=[crf_viterbi_accuracy]) #try adam optimizer

            self.model.summary()

    def train(self, data, batch_size, n_epochs, test_size=0.1):
        x_words_enc, x_stem_enc, x_char_enc, y_enc = self.__preprocess(data)
        return self.__train(x_words_enc, x_stem_enc, x_char_enc, y_enc, batch_size, n_epochs, test_size)

    def __preprocess(self, data):
        print("Number of sentences: ", len(data.groupby(['Sentence #'])))

        words = list(set(data["Word"].values))
        n_words = len(words)
        print("Number of words in the dataset: ", n_words)

        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        stems = list(set(stemmer.stem(word) for word in tqdm(words)))

        chars = set([w_i for w in tqdm(words) for w_i in w])
        print(chars)

        tags = list(set(data["Tag"].values))
        print("Tags:", tags)
        n_tags = len(tags)
        print("Number of Tags: ", n_tags)

        self.wordIndex.add(words)
        self.stemIndex.add(stems)
        self.tagIndex.add(tags)
        self.charIndex.add(chars)

        getter = SentenceGetter(data)

        sentences = getter.sentences

        encodedSentences = encodeSentences(sentences, self.wordIndex)
        encodedSentences = pad(encodedSentences, self.maxLengthSentence, self.wordIndex.getPadIdx())

        encodedStems = encodeStems(sentences, self.stemIndex, stemmer)
        encodedStems = pad(encodedStems, self.maxLengthSentence, self.stemIndex.getPadIdx())

        encodedTags = encodeTags(sentences, self.tagIndex)
        encodedTags = pad(encodedTags, self.maxLengthSentence, self.tagIndex.getPadIdx())
        encodedTags = onehotEncodeTags(encodedTags, n_tags)

        encodedChars = encodeChars(sentences, self.charIndex, self.maxLengthSentence, self.maxLengthWord)

        return  (encodedSentences, encodedStems, encodedChars, encodedTags)

    def __train(self, xWords, xStems, xChars, y, batchSize, epochs, testSize):
        self.X_word_tr, self.X_word_te, self.y_tr, self.y_te = train_test_split(xWords, y, test_size=testSize, random_state=2018)
        self.X_stem_tr, self.X_stem_te, _, _ = train_test_split(xStems, y, test_size=testSize, random_state=2018)
        self.X_char_tr, self.X_char_te, _, _ = train_test_split(xChars, y, test_size=testSize, random_state=2018)
        self.X_te = [
                        self.X_word_te,
                        self.X_stem_te,
                        np.array(self.X_char_te).reshape(len(self.X_char_te), self.maxLengthSentence, self.maxLengthWord)
                    ]

        early_stopping = EarlyStopping(monitor='loss', min_delta=0.0030, patience=2, verbose=1)

        self.history = self.model.fit(
                                    [
                                        self.X_word_tr,
                                        self.X_stem_tr,
                                        np.array(self.X_char_tr).reshape((len(self.X_char_tr), self.maxLengthSentence, self.maxLengthWord))
                                    ],
                                    np.array(self.y_tr),
                                    batch_size=batchSize, 
                                    epochs=epochs,
                                    validation_data=(self.X_te, np.array(self.y_te)),
                                    callbacks=[early_stopping],
                                    verbose=2)

    def evaluate(self):
        if len(self.X_te[0]) == 0:
            return

        pred_cat = self.model.predict(self.X_te)
        pred = np.argmax(pred_cat, axis=-1)
        y_te_true = np.argmax(self.y_te, -1)

        # Convert the index to tag
        pred_tag = [[self.tagIndex.getWord(i) for i in row] for row in pred]
        y_te_true_tag = [[self.tagIndex.getWord(i) for i in row] for row in y_te_true] 

        report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
        print(report)

    def predict(self, sentence):
        #todo preprocess sentence
        p = self.model.predict([
                            np.array([inputWords[0]]),
                            np.array([inputStems[0]]),
                            np.array(inputChars[0]).reshape(1, self.maxLengthSentence, self.maxLengthWord)
                        ])
        p = np.argmax(p, axis=-1)
        return p
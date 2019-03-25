from NerModel import NerModel
from WordIndex import WordIndex
from SentenceGetter import SentenceGetter
from Preprocessor import encodeSentences, encodeLemmas, encodeChars, encodeTags, pad, onehotEncodeTags

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D, Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

import nltk
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report

import numpy as np

from tqdm import tqdm

class LemmaCharLstmCrfModel(NerModel):
    """Class representing an Char Embedding + LSTM CRF Model for NER"""
    def __init__(self, dimSentenceEmbedding=0, dimWordEmbedding=0, maxLengthSentence=0, maxLengthWord=0):
        super().__init__()

        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()

        self.maxLengthSentence = maxLengthSentence
        self.maxLengthWord = maxLengthWord
        self.dimSentenceEmbedding = dimSentenceEmbedding
        self.dimWordEmbedding = dimWordEmbedding

        #lazy init model after preprocessing training data
        #word sizes 10% +, chars, tags exact

    def train(self, data, batch_size, n_epochs, test_size=0.1):
        x_words_enc, x_lemma_enc, x_char_enc, y_enc = self.__preprocess(data)
        if self.model == None:
            self.__init_model()
        return self.__train(x_words_enc, x_lemma_enc, x_char_enc, y_enc, batch_size, n_epochs, test_size)

    def __init_model(self):
        if (self.nWords > 0 and self.nTags > 0 and self.nChars > 0):
            # Model definition
            inputWords = Input(shape=(self.maxLengthSentence,))
            embeddingWords = Embedding(input_dim=self.nWords+2, output_dim=self.dimSentenceEmbedding, # n_words + 2 (PAD & UNK)
                            input_length=self.maxLengthSentence, mask_zero=True)(inputWords)  # default: 20-dim embedding

            inputLemmas = Input(shape=(self.maxLengthSentence,))
            embeddingLemmas = Embedding(input_dim=self.nWords+2, output_dim=self.dimSentenceEmbedding, # n_words + 2 (PAD & UNK)
                            input_length=self.maxLengthSentence, mask_zero=True)(inputLemmas)  # default: 20-dim embedding

            inputChars = Input(shape=(self.maxLengthSentence, self.maxLengthWord,))
            embeddingChars = TimeDistributed(Embedding(input_dim=self.nChars + 2, output_dim=self.dimWordEmbedding, # n_chars + 2 (PAD & UNK)
                            input_length=self.maxLengthWord, mask_zero=False))(inputChars)

            charDropout = Dropout(0.5)(embeddingChars)
            charCnn = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(charDropout) #convolution over characters; faster than lstm
            maxpoolOut = TimeDistributed(MaxPooling1D(self.maxLengthWord), name="Maxpool")(charCnn)
            charCnnOut = TimeDistributed(Flatten(), name="Flatten")(maxpoolOut)
            charCnnOut = Dropout(0.6)(charCnnOut)
            
            embeddingCombined = SpatialDropout1D(0.4)(concatenate([embeddingWords, embeddingLemmas, charCnnOut]))

            mainLstmOne = Bidirectional(LSTM(units=70, return_sequences=True,
                                    recurrent_dropout=0.5))(embeddingCombined)  # variational biLSTM
            mainLstmTwo = Bidirectional(LSTM(units=70, return_sequences=True,
                                    recurrent_dropout=0.25))(mainLstmOne)

            nn = TimeDistributed(Dense(70, activation="relu"))(mainLstmTwo)  # a dense layer as suggested by neuralNer
            crf = CRF(self.nTags+1)  # CRF layer, n_tags+1(PAD)
            out = crf(nn)  # output

            self.model = Model([inputWords, inputLemmas, inputChars], out)
            self.model.compile(optimizer="nadam", loss=crf_loss, metrics=[crf_viterbi_accuracy]) #try adam optimizer

            self.model.summary()

    def __preprocess(self, data):
        print("Number of sentences: ", len(data.groupby(['Sentence #'])))

        words = list(set(data["Word"].values))
        self.nWords = len(words)
        print("Number of words in the dataset: ", self.nWords)
        self.nWords += int(self.nWords * 0.1)

        lemmas = list(set(self.lemmatizer.lemmatize(word) for word in tqdm(words)))

        chars = set([w_i for w in tqdm(words) for w_i in w])
        print(chars)
        self.nChars = len(chars)

        tags = list(set(data["Tag"].values))
        print("Tags:", tags)
        self.nTags = len(tags)
        print("Number of Tags: ", self.nTags)

        if self.model == None:
            self.wordIndex = WordIndex("UNK", self.nWords)
            self.lemmaIndex = WordIndex("UNK", self.nWords)
            self.tagIndex = WordIndex("O", self.nTags+2)
            self.charIndex = WordIndex("UNK", self.nChars+1)

        self.wordIndex.add(words)
        self.lemmaIndex.add(lemmas)
        self.tagIndex.add(tags)
        self.charIndex.add(chars)

        getter = SentenceGetter(data)

        sentences = getter.sentences

        encodedSentences = encodeSentences(sentences, self.wordIndex)
        encodedSentences = pad(encodedSentences, self.maxLengthSentence, self.wordIndex.getPadIdx())

        encodedLemmas = encodeLemmas(sentences, self.lemmaIndex, self.lemmatizer)
        encodedLemmas = pad(encodedLemmas, self.maxLengthSentence, self.lemmaIndex.getPadIdx())

        encodedTags = encodeTags(sentences, self.tagIndex)
        encodedTags = pad(encodedTags, self.maxLengthSentence, self.tagIndex.getPadIdx())
        encodedTags = onehotEncodeTags(encodedTags, self.nTags)

        encodedChars = encodeChars(sentences, self.charIndex, self.maxLengthSentence, self.maxLengthWord)

        return  (encodedSentences, encodedLemmas, encodedChars, encodedTags)

    def __train(self, xWords, xLemmas, xChars, y, batchSize, epochs, testSize):
        self.X_word_tr, self.X_word_te, self.y_tr, self.y_te = train_test_split(xWords, y, test_size=testSize, random_state=2018)
        self.X_lemma_tr, self.X_lemma_te, _, _ = train_test_split(xLemmas, y, test_size=testSize, random_state=2018)
        self.X_char_tr, self.X_char_te, _, _ = train_test_split(xChars, y, test_size=testSize, random_state=2018)
        self.X_te = [
                        self.X_word_te,
                        self.X_lemma_te,
                        np.array(self.X_char_te).reshape(len(self.X_char_te), self.maxLengthSentence, self.maxLengthWord)
                    ]

        early_stopping = EarlyStopping(monitor='loss', min_delta=0.0040, patience=2, verbose=1)

        self.history = self.model.fit(
                                    [
                                        self.X_word_tr,
                                        self.X_lemma_tr,
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
        words = nltk.pos_tag(nltk.word_tokenize(sentence))
        encodedInput = encodeSentences([words], self.wordIndex)
        encodedInput = pad(encodedInput, self.maxLengthSentence, self.wordIndex.getPadIdx())

        encodedLemmas = encodeLemmas([[[w[0], w[1]] for w in words]], self.lemmaIndex, self.lemmatizer)
        encodedLemmas = pad(encodedLemmas, self.maxLengthSentence, self.lemmaIndex.getPadIdx())

        encodedChars = encodeChars([words], self.charIndex, self.maxLengthSentence, self.maxLengthWord)


        prediction = self.model.predict([
                            np.array([encodedInput[0]]),
                            np.array([encodedLemmas[0]]),
                            np.array(encodedChars[0]).reshape(1, self.maxLengthSentence, self.maxLengthWord)
                        ])
        prediction = np.argmax(prediction, axis=-1)
        return zip(words, [self.tagIndex.getWord(p) for p in prediction[0]])
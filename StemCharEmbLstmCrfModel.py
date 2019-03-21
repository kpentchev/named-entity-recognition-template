import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate, SpatialDropout1D, Conv1D, Flatten, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model

class StemCharEmbLstmCrfModel(object):
    """Class representing an Char Embedding + LSTM CRF Model for NER"""
    def __init__(self, nWords=0, nChars=0, nTags=0, lenEmdSent=0, lenEmdWord=0, maxLengthSentence=0, maxLengthWord=0):
        
        if (nWords > 0 and nTags > 0 and nChars > 0):

            self.maxLengthSentence = maxLengthSentence
            self.maxLengthWord = maxLengthWord

            # Model definition
            inputWords = Input(shape=(maxLengthSentence,))
            embeddingWords = Embedding(input_dim=nWords+2, output_dim=lenEmdSent, # n_words + 2 (PAD & UNK)
                            input_length=maxLengthSentence, mask_zero=True)(inputWords)  # default: 20-dim embedding

            inputStems = Input(shape=(maxLengthSentence,))
            embeddingStems = Embedding(input_dim=nWords+2, output_dim=lenEmdSent, # n_words + 2 (PAD & UNK)
                            input_length=maxLengthSentence, mask_zero=True)(inputStems)  # default: 20-dim embedding
            #stemsLstm = Bidirectional(LSTM(units=50, return_sequences=True,
            #                        recurrent_dropout=0.5))(embeddingStems)

            inputChars = Input(shape=(maxLengthSentence, maxLengthWord,))
            embeddingChars = TimeDistributed(Embedding(input_dim=nChars + 2, output_dim=lenEmdWord, # n_chars + 2 (PAD & UNK)
                            input_length=maxLengthWord, mask_zero=False))(inputChars)
            #charLstm = TimeDistributed(Bidirectional(LSTM(units=30, return_sequences=False, recurrent_dropout=0.6)))(embeddingChars)

            charDropout = Dropout(0.5)(embeddingChars)
            charCnn = TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same', activation='tanh', strides=1))(charDropout)
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

    def train(self, xWords, xStems, xChars, y, batchSize, epochs, testSize=0.0):
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

    def evaluate(self, wordIndex, idx2tag):
        if len(self.X_te[0]) == 0:
            return

        pred_cat = self.model.predict(self.X_te)
        pred = np.argmax(pred_cat, axis=-1)
        y_te_true = np.argmax(self.y_te, -1)

        # Convert the index to tag
        pred_tag = [[idx2tag.getWord(i) for i in row] for row in pred]
        y_te_true_tag = [[idx2tag.getWord(i) for i in row] for row in y_te_true] 

        report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
        print(report)

        '''
        i = np.random.randint(0,self.X_word_te.shape[0]) # choose a random number between 0 and len(X_te)
        p = self.model.predict(np.array([self.X_word_te[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(self.y_te[i], -1)

        print("Sample number {} of {} (Test Set)".format(i, self.X_word_te.shape[0]))
        # Visualization
        print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
        print(30 * "=")
        for w, t, pred in zip(self.X_word_te[i], true, p[0]):
            if w != 0:
                print("{:15}: {:5} {}".format(wordIndex.getWord(w), idx2tag.getWord(t), idx2tag.getWord(pred)))
        '''

    def predict(self, inputWords, inputStems, inputChars):
        p = self.model.predict([
                            np.array([inputWords[0]]),
                            np.array([inputStems[0]]),
                            np.array(inputChars[0]).reshape(1, self.maxLengthSentence, self.maxLengthWord)
                        ])
        p = np.argmax(p, axis=-1)
        return p

    def save(self, file):
        self.model.save(file)
        with open(file + '.params', 'wb') as handle:
            pickle.dump([self.maxLengthSentence, self.maxLengthWord], handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore(file):
    instance = StemCharEmbLstmCrfModel()
    with open(file + '.params', 'rb') as handle:
        params = pickle.load(handle)
        instance.maxLengthSentence = params[0]
        instance.maxLengthWord = params[1]
    instance.model = load_model(file, custom_objects={
        'CRF': CRF, 
        'crf_loss': crf_loss, 
        'crf_viterbi_accuracy': crf_viterbi_accuracy
    })
    instance.model.summary()
    return instance
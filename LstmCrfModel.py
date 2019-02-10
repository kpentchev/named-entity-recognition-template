import numpy as np
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.metrics import flat_classification_report
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras.models import load_model

class LstmCrfModel(object):
    """Class representing an LSTM CRF Model for NER"""
    def __init__(self, nWords=0, nTags=0, embedding=0, maxLength=0):
        
        if(nWords > 0 & nTags > 0):
            # Model definition
            input = Input(shape=(maxLength,))
            model = Embedding(input_dim=nWords+2, output_dim=embedding, # n_words + 2 (PAD & UNK)
                            input_length=maxLength, mask_zero=True)(input)  # default: 20-dim embedding
            model = Bidirectional(LSTM(units=50, return_sequences=True,
                                    recurrent_dropout=0.1))(model)  # variational biLSTM
            model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
            crf = CRF(nTags+1)  # CRF layer, n_tags+1(PAD)
            out = crf(model)  # output

            self.model = Model(input, out)
            self.model.compile(optimizer="rmsprop", loss=crf_loss, metrics=[crf_viterbi_accuracy])

            self.model.summary()

    def train(self, X, y, batchSize, epochs):
        self.X_tr, self.X_te, self.y_tr, self.y_te = train_test_split(X, y, test_size=0.1)
        self.X_tr.shape, self.X_te.shape, np.array(self.y_tr).shape, np.array(self.y_te).shape

        self.history = self.model.fit(self.X_tr, np.array(self.y_tr), batch_size=batchSize, epochs=epochs,
                    validation_split=0.1, verbose=2)

    def evaluate(self, words, idx2word, idx2tag):
        pred_cat = self.model.predict(self.X_te)
        pred = np.argmax(pred_cat, axis=-1)
        y_te_true = np.argmax(self.y_te, -1)

        # Convert the index to tag
        pred_tag = [[idx2tag[i] for i in row] for row in pred]
        y_te_true_tag = [[idx2tag[i] for i in row] for row in y_te_true] 

        report = flat_classification_report(y_pred=pred_tag, y_true=y_te_true_tag)
        print(report)

        i = np.random.randint(0,self.X_te.shape[0]) # choose a random number between 0 and len(X_te)
        p = self.model.predict(np.array([self.X_te[i]]))
        p = np.argmax(p, axis=-1)
        true = np.argmax(self.y_te[i], -1)

        print("Sample number {} of {} (Test Set)".format(i, self.X_te.shape[0]))
        # Visualization
        print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
        print(30 * "=")
        for w, t, pred in zip(self.X_te[i], true, p[0]):
            if w != 0:
                print("{:15}: {:5} {}".format(words[w-2], idx2tag[t], idx2tag[pred]))

    def predict(self, input):
        p = self.model.predict(np.array([input[0]]))
        p = np.argmax(p, axis=-1)
        return p

    def save(self, file):
        self.model.save(file)

def restore(file):
    instance = LstmCrfModel()
    instance.model = load_model(file, custom_objects={
        'CRF': CRF, 
        'crf_loss': crf_loss, 
        'crf_viterbi_accuracy': crf_viterbi_accuracy
    })
    instance.model.summary()
    return instance
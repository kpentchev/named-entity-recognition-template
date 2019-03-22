import pickle
from keras.models import load_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

class NerModel(object):
    """Class representing an abstract model"""
    def __init__(self):
        self.model = None
        self.params = {}

    def train(self, x_input, y_input, batch_size, n_epochs, test_size=0.1):
        raise NotImplementedError( "Model is an abstract class; method not implemented" )

    def evaluate(self):
        raise NotImplementedError( "Model is an abstract class; method not implemented" )

    def predict(self, input):
        raise NotImplementedError( "Model is an abstract class; method not implemented" )

    def save(self, file):
        self.model.save(file)
        with open(file + '.params', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Don't pickle baz
        del state["model"]
        del state["history"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None


def restore(file):
    instance = None
    with open(file + '.params', 'rb') as handle:
        instance = pickle.load(handle)
        instance.model = load_model(file, custom_objects={
            'CRF': CRF, 
            'crf_loss': crf_loss, 
            'crf_viterbi_accuracy': crf_viterbi_accuracy
        })
        instance.model.summary()
    return instance
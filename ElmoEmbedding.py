import keras.backend as K
from keras.engine.topology import Layer

import tensorflow_hub as hub
import tensorflow as tf

class ElmoEmbedding(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbedding, self).build(input_shape)

    def call(self, x, mask=None):
        lengths = K.cast(K.argmax(K.cast(K.equal(x, '--PAD--'), 'uint8')), 'int32')
        result = self.elmo(inputs=dict(tokens=x, sequence_len=lengths),
                      as_dict=True,
                      signature='tokens',
                      )['elmo']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape + (self.dimensions,)
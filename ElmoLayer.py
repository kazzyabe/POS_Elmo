import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.keras import backend as K

from tensorflow.keras.layers import Layer


class ElmoLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.batch_size = 32
        self.max_len = 50
        super(ElmoLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    
    def call(self, x, mask=None):
        return self.elmo(inputs={
                                    "tokens": tf.squeeze(tf.cast(x, tf.string)),
                                    "sequence_len": tf.constant(self.batch_size*[self.max_len])
                            },
                            signature="tokens",
                            as_dict=True)["elmo"]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, "__PAD__")


# class ElmoEmbeddingLayer(Layer):
#     def __init__(self, **kwargs):
#         self.dimensions = 1024
#         self.trainable=True
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
#                                name="{}_module".format(self.name))

#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)

#     def call(self, x, mask=None):
#         result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                       as_dict=True,
#                       signature='default',
#                       )['default']
#         return result

#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '--PAD--')

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.dimensions)
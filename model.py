# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dense, Dropout, Activation

import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.keras import backend as K
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda

import os


class POS_Tagger:
    def __init__(self, X_train=None, y_train=None, X_val=None, y_val=None, hidden_neurons=512, epochs=1, batch_size=32, verbose=1, max_len=50, n_tags=17, load_f=False, loadFile="tmp/model.h5"):
        '''
        load_f : flag to load a model (eg set to True load a model)
        '''
        tf.disable_eager_execution()

        self.X_train = X_train
        self.y_train = y_train
        self.val = (X_val, y_val)

        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        # session
        sess = tf.Session()
        K.set_session(sess)

        elmo_model = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        def ElmoEmbedding(x):
            return elmo_model(inputs={
                                    "tokens": tf.squeeze(tf.cast(x, tf.string)),
                                    "sequence_len": tf.constant(batch_size*[max_len])
                            },
                            signature="tokens",
                            as_dict=True)["elmo"]

        if load_f:
            self.model = load_model(loadFile)
        else:
            input_text = Input(shape=(max_len,), dtype=tf.string)
            embedding = Lambda(ElmoEmbedding, output_shape=(None, 1024))(input_text)
            x = Bidirectional(LSTM(units=hidden_neurons, return_sequences=True,
                                recurrent_dropout=0.2, dropout=0.2))(embedding)
            x_rnn = Bidirectional(LSTM(units=hidden_neurons, return_sequences=True,
                                    recurrent_dropout=0.2, dropout=0.2))(x)
            x = add([x, x_rnn])  # residual connection to the first biLSTM
            out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

            self.model = Model(input_text, out)
            self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            # self.model = Sequential([
            #     Dense(hidden_neurons, input_dim=X_train.shape[1]),
            #     Activation('relu'),
            #     Dropout(0.2),
            #     Dense(hidden_neurons),
            #     Activation('relu'),
            #     Dropout(0.2),
            #     Dense(y_train.shape[1], activation='softmax')
            # ])
            # self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, data=False, X_train=None, y_train=None, validation_data=None):
        checkpoint_path = "trained/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1)
        if not data:
            X_train = self.X_train
        if not data:
            y_train=self.y_train
        if not data:
            validation_data = self.val
        return self.model.fit(x=X_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=validation_data, callbacks=[cp_callback])
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)
    def save(self, f_name="tmp/model.h5"):
        self.model.save(f_name)

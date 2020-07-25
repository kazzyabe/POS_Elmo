import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.keras import backend as K
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Layer
from tensorflow.keras.metrics import Precision, Recall

from ElmoLayer import ElmoLayer

import os

class POS_Tagger:
    def __init__(self, hidden_neurons=512, epochs=1, batch_size=32, verbose=1, max_len=50, n_tags=17, load_f=False, loadFile="tmp/model.h5"):
        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        sess = tf.Session()
        K.set_session(sess)

        input_text = Input(shape=(max_len,), dtype=tf.string)
        # embed = ElmoLayer()
        # embedding = embed(input_text)
        embedding = ElmoLayer()(input_text)

        # Don't know why but it needs initialization after ElmoLayer
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        # print_emb = tf.Print(embedding, [embedding])
        x = Bidirectional(LSTM(units=hidden_neurons, return_sequences=True,
                            recurrent_dropout=0.2, dropout=0.2))(embedding)
        # x = Bidirectional(LSTM(units=hidden_neurons, return_sequences=True,
                            # recurrent_dropout=0.2, dropout=0.2))(print_emb)
        x_rnn = Bidirectional(LSTM(units=hidden_neurons, return_sequences=True,
                                recurrent_dropout=0.2, dropout=0.2))(x)
        x = add([x, x_rnn])  # residual connection to the first biLSTM
        out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

        self.model = Model(input_text, out)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    def fit(self, X_tr, y_tr, val):
        checkpoint_path = "trained/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        return self.model.fit(x=X_tr, y=y_tr, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=val, callbacks=[cp_callback])
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)

    def load(self, checkpoint_path):
        return self.model.load_weights(checkpoint_path)

    def predict(self, x):
        return self.model.predict(x)
    
    def save(self, checkpoint="trained/model.ckpt"):
        return self.model.save_weights(checkpoint)
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.keras import backend as K
import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import add
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Layer
from tensorflow.keras.metrics import Precision, Recall

import os

class POS_Tagger(Layer):
    
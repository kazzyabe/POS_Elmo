import tensorflow as tf

checkpoint_dir = "trained"
latest = tf.train.latest_checkpoint(checkpoint_dir)

from model import POS_Tagger

tagger = POS_Tagger()

import pickle as p
X_test = p.load(open("data/X_test.p", "rb"))
tagToIndex = p.load(open("data/tagToIndex.p", "rb"))

res = tagger.model.predict(X_test)
import numpy as np
m_res = np.argmax(res, axis=2)

IndexTotag = {}
for k in tagToIndex.keys():
    IndexTotag[tagToIndex[k]] = k

conv = lambda i: IndexTotag[i]
conv_vec = np.vectorize(conv)
conv_vec(m_res)
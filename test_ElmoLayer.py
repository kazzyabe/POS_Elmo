import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from tensorflow.compat.v1.keras import backend as K
from ElmoLayer import ElmoLayer
from tensorflow.keras import Model, Input


tf.disable_eager_execution()
sess = tf.Session()
K.set_session(sess)

max_len = 50
input_text = Input(shape=(max_len,), dtype=tf.string)
embed = ElmoLayer()
embeddings = embed(input_text)
mask = embed.compute_mask(input_text)
model = Model(input_text, embed)
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

import pickle as p
X_tr = p.load(open("data/X_train.p", "rb"))

res = model.predict(X_tr[0:32])
print(res)
import tensorflow.compat.v1 as tf

checkpoint_dir = "trained"
latest = tf.train.latest_checkpoint(checkpoint_dir)

from POS_Tagger import POS_Tagger

tf.disable_eager_execution()

tagger = POS_Tagger()
tagger.load(latest)

import pickle as p
X_tr = p.load(open("data/X_train.p", "rb"))
X_test = p.load(open("data/X_test.p", "rb"))
tagToIndex = p.load(open("data/tagToIndex.p", "rb"))

res_tr = tagger.model.predict(X_tr)
res_test = tagger.model.predict(X_test)

p.dump(res_tr, open("Res/X_train.p", "wb"))
p.dump(res_test, open("Res/X_test.p", "wb"))




# IndexTotag = {}
# for k in tagToIndex.keys():
#     IndexTotag[tagToIndex[k]] = k

# conv = lambda i: IndexTotag[i]
# conv_vec = np.vectorize(conv)
# conv_vec(m_res)
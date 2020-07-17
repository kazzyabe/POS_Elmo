from model import POS_Tagger

batch_size = 32

import pickle as p
X_tr = p.load(open("X_tr.p", "rb"))
X_te = p.load(open("X_te.p", "rb"))
y_tr = p.load(open("y_tr.p", "rb"))
y_te = p.load(open("y_te.p", "rb"))


import numpy as np
X_tr, X_val = np.array(X_tr[:121*batch_size]), np.array(X_tr[-13*batch_size:])
y_tr, y_val = y_tr[:121*batch_size], y_tr[-13*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)


tagger = POS_Tagger()
tagger.fit(data=True, X_train=X_tr, y_train=y_tr, validation_data=(X_val, y_val))
tagger.save()
import pickle as p
X_tr = p.load(open("data/X_train.p", "rb"))
X_test = p.load(open("data/X_test.p", "rb"))
res_tr = p.load(open("Res/X_train.p", "rb"))
res_test = p.load(open("Res/X_test.p", "rb"))
y_train = p.load(open("data/y_train.p", "rb"))
y_test = p.load(open("data/y_test.p", "rb"))

import numpy as np
res_tr = np.argmax(res_tr, axis=2)
res_test = np.argmax(res_test, axis=2)

y_tr = y_train.reshape(y_train.shape[0], y_train.shape[1])
y_te = y_test.reshape(y_test.shape[0], y_test.shape[1])

tr_G = []
te_G = []
tr_P = []
te_P = []

for i in range(X_tr.shape[0]):
    for j in range(X_tr.shape[1]):
        if X_tr[i,j] == '__PAD__':
            break
        else:
            tr_G.append(y_tr[i, j])
            tr_P.append(res_tr[i,j])

for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        if X_test[i,j] == '__PAD__':
            break
        else:
            te_G.append(y_te[i,j])
            te_P.append(res_test[i,j])

from sklearn.metrics import f1_score
macro_tr = f1_score(tr_G, tr_P, average='macro')
macro_te = f1_score(te_G, te_P, average='macro')

micro_tr = f1_score(tr_G, tr_P, average='micro')
micro_te = f1_score(te_G, te_P, average='micro')

weighted_tr = f1_score(tr_G, tr_P, average='weighted')
weighted_te = f1_score(te_G, te_P, average='weighted')

# samples_tr = f1_score(tr_G, tr_P, average='samples')
# samples_te = f1_score(tr_G, tr_P, average='samples')

print("Train")
print("macro = ", macro_tr)
print("micro = ", micro_tr)
print("weighted = ", weighted_tr)
print("Test")
print("macro = ", macro_te)
print("micro = ", micro_te)
print("weighted = ", weighted_te)
from copy import deepcopy
import pickle as p

# def preprocessed_data(d_name):
#     '''Return X and Y train test val'''
    # Make the predictable random
import numpy as np

CUST_SEED = 50
np.random.seed(CUST_SEED)

# Data set from nltk: (term, tag)
from UD_converter import convertUD
d_name = "/Users/kazuyabe/Data/UD_English-EWT"
sentences = convertUD(d_name)

# Extracting a set of tags: 12 tags
tags = set([tag for sent in sentences for _, tag in sent])
# Extracting a set of words: 12408 words
words = set([term for sent in sentences for term,_ in sent])

### Separating into train, validation, test data sets
train_test_cutoff = int(.80 * len(sentences)) 
training_sentences = sentences[:train_test_cutoff]
testing_sentences = sentences[train_test_cutoff:]
train_val_cutoff = int(.25 * len(training_sentences))
validation_sentences = training_sentences[:train_val_cutoff]
training_sentences = training_sentences[train_val_cutoff:]

def transform_to_dataset(tagged_sentences):
    """
    Split tagged sentences to X and y datasets and append some basic features.
:param tagged_sentences: a list of POS tagged sentences
    :param tagged_sentences: list of list of tuples (term_i, tag_i)
    :return: 
    """
    X, y = [], []
    for pos_tags in tagged_sentences:
        sents_tmp = []
        tags_tmp = []
        for (term, class_) in pos_tags:
            # Add basic NLP features for each sentence term
            # X.append(add_basic_features(untag(pos_tags), index))
            sents_tmp.append(term)
            tags_tmp.append(class_)
        X.append(deepcopy(sents_tmp))
        y.append(deepcopy(tags_tmp))
    return X, y

# X  [sent, sent, ...]: sent = [word, word, ...]
# y  [sent, sent, ...]: sent = [pos, pos, ...]
X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(testing_sentences)
X_val, y_val = transform_to_dataset(validation_sentences)

# converting y to index
tagToIndex = {t: i for i, t in enumerate(tags)}

def yToIndex(y):
    for i in range(len(y)):
        for j in range(len(y[i])):
            y[i][j] = tagToIndex[y[i][j]]
    return y

y_train = yToIndex(y_train)
y_test = yToIndex(y_test)
y_val = yToIndex(y_val)


##### Padding ###############################################
maxlen_train = [len(s) for s in X_train]
maxlen_train.sort()
maxlen_train = maxlen_train[-1]

maxlen_test = [len(s) for s in X_test]
maxlen_test.sort()
maxlen_test = maxlen_test[-1]

maxlen_val = [len(s) for s in X_val]
maxlen_val.sort()
maxlen_val = maxlen_val[-1]

def pad_X(X, ml):
    padded_X = []
    for s in X:
        if len(s) < ml:
            pad = ["__PAD__"] * (ml - len(s))
            padded_X.append(s+pad)
        else:
            if len(s) > ml:
                padded_X.append(s[:ml])
            else:
                padded_X.append(s)
    return padded_X

maxlen_train = 50

X_train = pad_X(X_train, maxlen_train)
X_test = pad_X(X_test, maxlen_train)
X_val = pad_X(X_val, maxlen_train)

# counting tags frequency
# tag_c = {}
# for y in y_train:
#     for w in y:
#         tag_c.setdefault(w, 0)
#         tag_c[w] += 1

## Padding y by "noun"
from tensorflow.keras.preprocessing.sequence import pad_sequences
y_train = pad_sequences(maxlen=maxlen_train, sequences=y_train, padding="post", value=tagToIndex["NOUN"])
y_val = pad_sequences(maxlen=maxlen_train, sequences=y_val, padding="post", value=tagToIndex["NOUN"])
y_test = pad_sequences(maxlen=maxlen_train, sequences=y_test, padding="post", value=tagToIndex["NOUN"])


import numpy as np
batch_size = 32
X_train, X_val = np.array(X_train[:121*batch_size]), np.array(X_val[-13*batch_size:])
y_train, y_val = y_train[:121*batch_size], y_val[-13*batch_size:]
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

p.dump(X_train, open("data/X_train.p", 'wb'))
p.dump(X_test, open("data/X_test.p", 'wb'))
p.dump(X_val, open("data/X_val.p", 'wb'))
p.dump(y_train, open("data/y_train.p", 'wb'))
p.dump(y_test, open("data/y_test.p", 'wb'))
p.dump(y_val, open("data/y_val.p", 'wb'))
p.dump(tagToIndex, open("data/tagToIndex.p", "wb"))

# X_train = p.load(open("data/X_train.p", "rb"))
# X_test = p.load(open("data/X_test.p", "rb"))
# X_val = p.load(open("data/X_val.p", "rb"))
# y_train = p.load(open("data/y_train.p", "rb"))
# y_test = p.load(open("data/y_test.p", "rb"))
# y_val = p.load(open("data/y_val.p", "rb"))
# tagToIndex = p.load(open("data/tagToIndex.p", "rb"))

# # Encode POS to int
# from sklearn.preprocessing import LabelEncoder
# # Fit LabelEncoder with our list of classes
# label_encoder = LabelEncoder()
# label_encoder.fit(y_train + y_test + y_val)
# # Encode class values as integers
# y_train = label_encoder.transform(y_train)
# y_test = label_encoder.transform(y_test)
# y_val = label_encoder.transform(y_val)

# Convert integers to dummy variables (one hot encoded)
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# y_val = to_categorical(y_val)

# return X_train, X_test, X_val, y_train, y_test, y_val
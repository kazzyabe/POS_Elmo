import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="path to the Universal Dependencies data directory", default="data/UD_Japanese-GSD")
parser.add_argument("-m", "--model", help="name for saving the model", default="./tmp/model.h5")
args = parser.parse_args()

# Getting data
from preprocessing import preprocessed_data
X_train, X_test, X_val, y_train, y_test, y_val = preprocessed_data(args.directory)

from tensorflow.keras.utils import plot_model
from tensorflow import keras

clf = keras.models.load_model('tmp/model.h5')

score = clf.score(X_test, y_test)
print(score)
plot_model(clf.model, to_file='model.png', show_shapes=True)

# mple - loss: 0.2152 - accuracy: 0.9351
# [0.23645427182001502, 0.93505]
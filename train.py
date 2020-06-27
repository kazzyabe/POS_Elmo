import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", help="path to the Universal Dependencies data directory", default="/Users/kazuyabe/Data/UD_Japanese-GSD")
parser.add_argument("-m", "--model", help="name for saving the model", default="./tmp/model.h5")
args = parser.parse_args()

# Getting data
from preprocessing import preprocessed_data
X_train, X_test, X_val, y_train, y_test, y_val = preprocessed_data(args.directory)

# Get model
# from model import wrapped_model
# clf = wrapped_model(X_train, y_train, X_val, y_val)
from model import POS_Tagger
tagger = POS_Tagger(X_train, y_train, X_val, y_val, epochs=2)

# Training
hist = tagger.fit()

tagger.save(args.model)

# from plot import plot_model_performance
# plot_model_performance(
#     train_loss=hist.history.get('loss', []),
#     train_acc=hist.history.get('acc', []),
#     train_val_loss=hist.history.get('val_loss', []),
#     train_val_acc=hist.history.get('val_acc', [])
# )

score = tagger.evaluate(X_test, y_test)
print(score)

# from tensorflow.keras.utils import plot_model
# plot_model(clf.model, to_file='model.png', show_shapes=True)

# clf.model.save('/tmp/model.h5')


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation
# def build_model(input_dim, hidden_neurons=512, output_dim):
#     """
#     Construct, compile and return a Keras model which will be used to fit/predict
#     """
#     model = Sequential([
#         Dense(hidden_neurons, input_dim=input_dim),
#         Activation('relu'),
#         Dropout(0.2),
#         Dense(hidden_neurons),
#         Activation('relu'),
#         Dropout(0.2),
#         Dense(output_dim, activation='softmax')
#     ])
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model


# # Wrap the model to use scikit-learn classifier
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# def wrapped_model(X_train, y_train, X_val, y_val):
#     model_params = {
#         'build_fn': build_model,
#         'input_dim': X_train.shape[1],
#         'hidden_neurons': 512,
#         'output_dim': y_train.shape[1],
#         'epochs': 5,
#         'batch_size': 256,
#         'verbose': 1,
#         'validation_data': (X_val, y_val),
#         'shuffle': True
#     }
#     return KerasClassifier(**model_params)

class POS_Tagger:
    def __init__(self, X_train=None, y_train=None, X_val=None, y_val=None, hidden_neurons=512, epochs=5, batch_size=256, verbose=1, load_f=False, loadFile="tmp/model.h5"):
        '''
        load_f : flag to load a model (eg set to True load a model)
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.val = (X_val, y_val)

        self.hidden_neurons = hidden_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

        if load_f:
            self.model = load_model(loadFile)
        else:
            self.model = Sequential([
                Dense(hidden_neurons, input_dim=X_train.shape[1]),
                Activation('relu'),
                Dropout(0.2),
                Dense(hidden_neurons),
                Activation('relu'),
                Dropout(0.2),
                Dense(y_train.shape[1], activation='softmax')
            ])
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    def fit(self, X_train=None, y_train=None, validation_data=None):
        if not X_train:
            X_train = self.X_train
        if not y_train:
            y_train=self.y_train
        if not validation_data:
            validation_data = self.val
        return self.model.fit(x=X_train, y=y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=self.verbose, validation_data=validation_data)
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(x=X_test, y=y_test, batch_size=self.batch_size, verbose=self.verbose)
    def save(self, f_name="tmp/model.h5"):
        self.model.save(f_name)

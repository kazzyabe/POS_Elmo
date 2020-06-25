from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
def build_model(input_dim, hidden_neurons, output_dim):
    """
    Construct, compile and return a Keras model which will be used to fit/predict
    """
    model = Sequential([
        Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Wrap the model to use scikit-learn classifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def wrapped_model(X_train, y_train, X_val, y_val):
    model_params = {
        'build_fn': build_model,
        'input_dim': X_train.shape[1],
        'hidden_neurons': 512,
        'output_dim': y_train.shape[1],
        'epochs': 5,
        'batch_size': 256,
        'verbose': 1,
        'validation_data': (X_val, y_val),
        'shuffle': True
    }
    return KerasClassifier(**model_params)
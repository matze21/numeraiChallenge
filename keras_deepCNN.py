from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model

def deepNN(n_inputFeatures):

    X_input = Input(shape=(n_inputFeatures,))

    X = Dense(n_inputFeatures, activation='relu')(X_input)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(n_inputFeatures, activation='relu')(X)
    X = Dense(1, activation="sigmoid")(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

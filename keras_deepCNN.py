from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model

def deepNN(n_inputFeatures):

    X_input = Input(shape=(n_inputFeatures,))

    X = Dense(n_inputFeatures, activation='relu')(X_input)
    X = BatchNormalization(axis = -1)(X)
    # X = Dense(n_inputFeatures*5, activation='relu')(X)
    # X = BatchNormalization(axis = -1)(X)
    # X = Dense(n_inputFeatures*3, activation='relu')(X)
    # X = BatchNormalization(axis = -1)(X)
    # X = Dense(n_inputFeatures*2, activation='relu')(X)
    # X = BatchNormalization(axis = -1)(X)
    X = Dense(250, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(250, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(200, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(200, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(150, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(100, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(80, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(60, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(50, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(40, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(30, activation='relu')(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(1, activation="sigmoid")(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

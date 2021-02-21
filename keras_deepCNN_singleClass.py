from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers

regularizationConst_l1 = 0.0005
regularizationConst_l2 = 0.0005

def deepNN(n_inputFeatures):

    X_input = Input(shape=(n_inputFeatures,))
    X = BatchNormalization(axis = -1)(X_input)
    X = Dense(n_inputFeatures, activation='relu', 
    kernel_regularizer=regularizers.l1_l2(l1=3e-5, l2=4e-4),
    bias_regularizer=regularizers.l2(5e-4),
    activity_regularizer=regularizers.l2(1e-5))(X)
    
    size = 254
    dropoutRate = 0.3
    X = Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=3e-5, l2=9e-4), bias_regularizer=regularizers.l2(9e-4), activity_regularizer=regularizers.l2(2e-5))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=3e-5, l2=9e-4), bias_regularizer=regularizers.l2(9e-4), activity_regularizer=regularizers.l2(2e-5))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=3e-5, l2=9e-4), bias_regularizer=regularizers.l2(9e-4), activity_regularizer=regularizers.l2(2e-5))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)

    X = Dense(1, activation="sigmoid")(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

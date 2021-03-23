from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers

regularizationConst_l1 = 0#0.0002
regularizationConst_l2 = 0#0.0003

def deepNN(n_inputFeatures):

    X_input = Input(shape=(n_inputFeatures,))
    #X = BatchNormalization(axis = -1)(X_input)
    # X = Dense(n_inputFeatures, activation='relu', 
    # kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2),
    # bias_regularizer=regularizers.l2(regularizationConst_l2),
    # activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    
    size = 1024
    dropoutRate = 0.2
    X = Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X_input)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    # X = Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    # X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size/2, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size/4, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size/8, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size/16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(size/32, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(5, activation="softmax")(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

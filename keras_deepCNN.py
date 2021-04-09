from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers

def deepNN(n_inputFeatures):
    regularizationConst_l1 = 0.00002
    regularizationConst_l2 = 0.00001
    activation = "relu"
    #size = 512
    dropoutRate = 0.2
    X_input = Input(shape=(n_inputFeatures,))
    X = Dropout(0.9, input_shape = (n_inputFeatures,))(X_input)
    #X = BatchNormalization(axis = -1)(X_input)
    # X = Dense(n_inputFeatures, activation=activation, 
    # kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2),
    # bias_regularizer=regularizers.l2(regularizationConst_l2),
    # activity_regularizer=regularizers.l2(regularizationConst_l2))(X)

    
    X = Dense(512, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(512, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = BatchNormalization(axis = -1)(X)
    X = Dense(512, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = BatchNormalization(axis = -1)(X)



    # X = Dense(size/2, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    # X = Dropout(dropoutRate, input_shape = (size,))(X)
    # X = BatchNormalization(axis = -1)(X)
    # X = Dense(size/4, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    # X = Dropout(dropoutRate, input_shape = (size,))(X)
    # X = BatchNormalization(axis = -1)(X)
    # X = Dense(size/8, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    # X = Dropout(dropoutRate, input_shape = (size,))(X)
    # X = BatchNormalization(axis = -1)(X)
    # X = Dense(size/16, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    # X = Dropout(dropoutRate, input_shape = (size,))(X)
    # X = BatchNormalization(axis = -1)(X)
    # X = Dense(size/32, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    # X = Dropout(dropoutRate, input_shape = (size,))(X)
    X = Dense(5, activation="softmax")(X)
    #X = Dense(1, activation="linear")(X)

    # Create model
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

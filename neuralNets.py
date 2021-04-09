import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import numpy as np

def defineNN_3classes(n_inputFeatures):
    activation = "relu"
    regularizationConst_l1 = 0.000#2
    regularizationConst_l2 = 0.000#1
    size = 512
    X_input = Input(shape=(n_inputFeatures,))
    X = Dropout(0.0, input_shape = (n_inputFeatures,))(X_input)
 
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    #X = BatchNormalization(axis = -1)(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    #X = BatchNormalization(axis = -1)(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    #X = BatchNormalization(axis = -1)(X)
    X = Dense(size/4, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size/8, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size/16, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size/32, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(3, activation="softmax")(X)
    
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

def defineNN_2classes(n_inputFeatures):
    activation = "relu"
    regularizationConst_l1 = 0.00000#3
    regularizationConst_l2 = 0.00000#3
    size = 512
    X_input = Input(shape=(n_inputFeatures,))
    X = Dropout(0.00, input_shape = (n_inputFeatures,))(X_input)
 
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    #X = BatchNormalization(axis = -1)(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    #X = BatchNormalization(axis = -1)(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)

    #X = Dropout(dropoutRate, input_shape = (size,))(X)
    #X = BatchNormalization(axis = -1)(X)
    X = Dense(size/4, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size/8, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size/16, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)
    X = Dense(size/32, activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularizationConst_l1, l2=regularizationConst_l2), bias_regularizer=regularizers.l2(regularizationConst_l2), activity_regularizer=regularizers.l2(regularizationConst_l2))(X)

    X = Dense(1, activation="sigmoid")(X)
    
    model = Model(inputs = X_input, outputs = X, name='deepNN')

    return model

def oneHotEncodeData3Classes(targets):
    j=0
    Y_val = np.zeros((targets.shape[0], 3))
    for j in range(targets.shape[0]):
        if targets[j] == 0:
            Y_val[j, 0] = 1
        elif targets[j] == 1:
            Y_val[j, 1] = 1
        elif targets[j] == 2:
            Y_val[j, 2] = 1
        else:
            print("something went wrong, new class", targets[j])
    return Y_val

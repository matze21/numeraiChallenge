import pandas as pd
#import numerapi
import sklearn.linear_model

import keras_deepCNN
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import time

validation_data = pd.read_csv("data/numerai_datasets_21.02.21/numerai_tournament_data.csv")
feature_cols = validation_data.columns[validation_data.columns.str.startswith('feature')]
trainingData_class0   = validation_data.loc[validation_data.target == 0]
trainingData_class025 = validation_data.loc[validation_data.target == 0.25]
trainingData_class05  = validation_data.loc[validation_data.target == 0.5]
trainingData_class075 = validation_data.loc[validation_data.target == 0.75]
trainingData_class1   = validation_data.loc[validation_data.target == 1.0]      
X_label_0   = trainingData_class0[feature_cols].reset_index().drop(['index'], axis = 1)
X_label_025 = trainingData_class025[feature_cols].reset_index().drop(['index'], axis = 1)
X_label_05  = trainingData_class05[feature_cols].reset_index().drop(['index'], axis = 1)
X_label_075 = trainingData_class075[feature_cols].reset_index().drop(['index'], axis = 1)
X_label_1   = trainingData_class1[feature_cols].reset_index().drop(['index'], axis = 1)
Y_label_0   = trainingData_class0.target.reset_index().drop(['index'], axis = 1)
Y_label_025 = trainingData_class025.target.reset_index().drop(['index'], axis = 1)
Y_label_05  = trainingData_class05.target.reset_index().drop(['index'], axis = 1)
Y_label_075 = trainingData_class075.target.reset_index().drop(['index'], axis = 1)
Y_label_1   = trainingData_class1.target.reset_index().drop(['index'], axis = 1)
minLength = np.min([len(Y_label_0), len(Y_label_025), len(Y_label_05), len(Y_label_075), len(Y_label_1)])       
Y_val_data = Y_label_0
Y_val_data = Y_val_data.append(Y_label_025, ignore_index = True)
Y_val_data = Y_val_data.append(Y_label_05, ignore_index = True)
Y_val_data = Y_val_data.append(Y_label_075, ignore_index = True)
Y_val_data = Y_val_data.append(Y_label_1, ignore_index = True)       
X_val = X_label_0
X_val = X_val.append(X_label_025, ignore_index = True)
X_val = X_val.append(X_label_05, ignore_index = True)
X_val = X_val.append(X_label_075, ignore_index = True)
X_val = X_val.append(X_label_1, ignore_index = True)  
j=0
Y_val_npArray = Y_val_data.to_numpy()
Y_val = np.zeros((Y_val_npArray.shape[0], 5))
for j in range(Y_val_npArray.shape[0]):
    if Y_val_npArray[j] == 0:
        Y_val[j, 0] = 1
    elif Y_val_npArray[j] == 0.25:
        Y_val[j, 1] = 1
    elif Y_val_npArray[j] == 0.5:
        Y_val[j, 2] = 1
    elif Y_val_npArray[j] == 0.75:
        Y_val[j, 3] = 1
    elif Y_val_npArray[j] == 1.0:
        Y_val[j, 4] = 1
    else:
        print("something went wrong, new class", Y_val_npArray[j])

model = keras_deepCNN.deepNN(X_val.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("model_98PerAcc.h5")
model.evaluate(X_val,Y_val)

predictions = model.predict(X_val)

i=0
predictionVektor = np.zeros((predictions.shape[0]))
for i in range(predictions.shape[0]):
    maxPos = np.argmax(predictions[i,:])
    if maxPos == 0:
        predictionVektor[i] = 0
    elif maxPos == 1:
        predictionVektor[i] = 0.25
    elif maxPos == 2:
        predictionVektor[i] = 0.5
    elif maxPos == 3:
        predictionVektor[i] = 0.75
    elif maxPos == 4:
        predictionVektor[i] = 1.0
    else:
        print("something went wrong, new class", Y_train_npArray[i])

dataFrame = pd.DataFrame(X_val, columns=feature_cols)
dataFrame = pd.concat([dataFrame, Y_val_data], axis = 1)
PredictionsDF = pd.DataFrame(predictionVektor, columns = ['predictions'])
dataFrame = pd.concat([dataFrame, PredictionsDF], axis = 1)

dataFrame.to_csv('evaluate_model.csv')
import pandas as pd
#import numerapi
import sklearn.linear_model

import keras_deepCNN_singleClass
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import time
import tensorflow as tf

useValidationData=True

tic = time.time()
#training_data = pd.read_csv("appendedTrainingData_test12021-01-31_2021-03-21.csv")
training_data = pd.read_csv("data/numerai_datasets_28.03.21/numerai_training_data.csv")

feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
X_train = training_data[feature_cols].reset_index().drop(['index'], axis = 1)


X_train_0 = X_train.to_numpy()
Y_train_0 = training_data.target
Y_train_0 = Y_train_0.replace(0.5, 2)
Y_train_0 = Y_train_0.replace([0.25, 0.5, 0.75, 1], 0)
Y_train_0 = Y_train_0.replace(2, 1)
Y_train_0 = Y_train_0.to_numpy()

if useValidationData:
    tournament_data = pd.read_csv("data/numerai_datasets_28.03.21/numerai_tournament_data.csv")
    validation_data = tournament_data.loc[tournament_data.data_type == 'validation']
    X_val = validation_data[feature_cols].reset_index().drop(['index'], axis = 1).to_numpy()
    Y_val = validation_data.target

    X_val_0 = X_val
    Y_val_0 = Y_val.replace(0.5, 2)
    Y_val_0 = Y_val_0.replace([0.25, 0.5, 0.75, 1], 0)
    Y_val_0 = Y_val_0.replace(2, 1)
    Y_val_0 = Y_val_0.to_numpy()


toc = time.time()
print("processed the data took ", toc - tic)

X_train_0, Y_train_0 = shuffle(X_train_0, Y_train_0)

model_0 = keras_deepCNN_singleClass.deepNN(X_train.shape[1])

#model_0.load_weights("model_singleClass.h5")
#opt = Adam(lr=0.00005, beta_1=0.9, beta_2=0.9, decay=1e-8)
model_0.compile(optimizer='adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall()])

history = model_0.fit(X_train_0, Y_train_0, epochs = 100, batch_size = 256*128, validation_data=(X_val_0, Y_val_0))
#import pdb; pdb.set_trace()

model_0.save_weights("model_singleClass2.h5")

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
print(history.history)
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


predictionVektorTrainData = model_0.predict(X_train_0)
dataFrame = pd.DataFrame(Y_train_0, columns=['target'])
dataFrame['predictions'] = predictionVektorTrainData
dataFrame.to_csv('evaluate_model_train.csv')

predictionVektorDevData = model_0.predict(X_val_0)
dataFrame = pd.DataFrame(Y_val_0, columns=['target'])
dataFrame['predictions'] = predictionVektorDevData
dataFrame.to_csv('evaluate_model_test.csv')



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
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

import tensorflow.keras as k
import time
import tensorflow as tf

def max_accuracy(y_true, y_pred):
  """Calculates how often the max prediction matches one-hot labels"""
  retVal = 0
  
  if y_true.shape[0] != None:  
    num_correct_classified = tf.math.argmax(y_true,axis = 1) == tf.math.argmax(y_pred, axis = 1)
    num = tf.reduce_sum(tf.dtypes.cast(num_correct_classified, tf.int32), axis = -1)
    retVal = num / y_true.shape[0]
  else:
    retVal = tf.dtypes.cast(tf.math.argmax(y_true,axis = 1) == tf.math.argmax(y_pred, axis = 1), tf.int32)
  return retVal

def oneHotEncodeData(targets):
    j=0
    Y_val = np.zeros((targets.shape[0], 5))
    for j in range(targets.shape[0]):
        if targets[j] == 0:
            Y_val[j, 0] = 1
        elif targets[j] == 0.25:
            Y_val[j, 1] = 1
        elif targets[j] == 0.5:
            Y_val[j, 2] = 1
        elif targets[j] == 0.75:
            Y_val[j, 3] = 1
        elif targets[j] == 1.0:
            Y_val[j, 4] = 1
        else:
            print("something went wrong, new class", targets[j])
    return Y_val

useValidationData=True

tic = time.time()
#training_data = pd.read_csv("balancedTrainingData_test12021-01-31_2021-03-21.csv")
training_data = pd.read_csv("data/numerai_datasets_28.03.21/numerai_training_data.csv")

training_data = training_data.drop_duplicates()

feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
X_train = training_data[feature_cols].to_numpy()
#Y_train = training_data.loc[:,['label_0', 'label_025', 'label_05', 'label_075', 'label_1']].to_numpy()
Y_train = training_data.target.to_numpy()

if useValidationData:
    tournament_data = pd.read_csv("data/numerai_datasets_28.03.21/numerai_tournament_data.csv")
    validation_data = tournament_data.loc[tournament_data.data_type == 'validation']
    X_val = validation_data[feature_cols].reset_index().drop(['index'], axis = 1).to_numpy()
    #X_pred = tournamend_data.loc[tournament_data.data_type == ""]
    Y_val = validation_data.target.to_numpy()



toc = time.time()
print("processed the data took ", toc - tic)

model = keras_deepCNN.deepNN(X_train.shape[1])

METRICS = [
    #   k.metrics.TruePositives(name='tp'),
    #   k.metrics.FalsePositives(name='fp'),
    #   k.metrics.TrueNegatives(name='tn'),
    #   k.metrics.FalseNegatives(name='fn'), 
    #   k.metrics.CategoricalAccuracy(name='cat_accuracy'),
    #   k.metrics.Precision(name='precision'),
    #   k.metrics.Recall(name='recall'),
    #   k.metrics.AUC(name='auc'),
    max_accuracy,
]

model.compile(optimizer='adam', loss='CategoricalCrossEntropy', metrics=METRICS)
model.load_weights("model.h5")

tic = time.time()
X_train, Y_train = shuffle(X_train, Y_train)
toc = time.time()
print("shuffle the data took ", toc - tic)

class_weights = class_weight.compute_class_weight('balanced', np.array([0, 0.25, 0.5, 0.75, 1]), Y_train)
class_weights = dict(enumerate(class_weights))

tic = time.time()
Y_train = oneHotEncodeData(Y_train)
Y_val = oneHotEncodeData(Y_val)
toc = time.time()
print("encode the data took ", toc - tic)

#import pdb; pdb.set_trace()

history = model.fit(X_train, Y_train, epochs = 8, batch_size = 256*128, class_weight=class_weights, validation_data=(X_val, Y_val))
#import pdb; pdb.set_trace()

model.save_weights("model.h5")

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



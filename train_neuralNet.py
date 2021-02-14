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

useValidationData=True

tic = time.time()
training_data = pd.read_csv("balancedTrainingData_test12021-01-31_2021-02-14.csv")

feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
X_train = training_data[feature_cols].to_numpy()
Y_train = training_data.loc[:,['label_0', 'label_025', 'label_05', 'label_075', 'label_1']].to_numpy()

toc = time.time()
print("loaded the data took ", toc - tic)

if useValidationData:
    validation_data = pd.read_csv("data/numerai_datasets_17.01.21/numerai_validation_data.csv")
    feature_cols_val = validation_data.columns[validation_data.columns.str.startswith('feature')]
    validation_features = validation_data[feature_cols_val]
    X_val = validation_features

    j=0
    Y_val_npArray = validation_data.target.to_numpy()
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
        #Y_val = validation_data.target
# find only the feature columns



toc = time.time()
print("processed the data took ", toc - tic)

model = keras_deepCNN.deepNN(X_train.shape[1])

opt = Adam(lr=0.00005, beta_1=0.9, beta_2=0.9, decay=1e-8)#learning_rates_dec[0])
#model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])
#model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='mae', metrics=['mean_squared_error'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.load_weights("model_moreData.h5")

X_train, Y_train = shuffle(X_train, Y_train)
n_valSet = 50000
X_val_test = X_train[0:n_valSet,:]
X_trainSet = X_train[n_valSet:X_train.shape[0]-1, :]

Y_val_test = Y_train[0:n_valSet,:]
Y_trainSet = Y_train[n_valSet:X_train.shape[0]-1,:]


history = model.fit(X_trainSet, Y_trainSet, epochs = 100, batch_size = 256*64*8*4, validation_data=(X_val_test, Y_val_test))
#import pdb; pdb.set_trace()

model.save_weights("model_moreData_100epochs.h5")

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

# model = sklearn.linear_model.LinearRegression(normalize = True)
# model.fit(training_features, training_data.target)
# predictions = model.predict(validation_features)
# predictions_df = validation_data["id"].to_frame()
# predictions_df["prediction_kazutsugi"] = predictions
# predictions_df.head()

# predictions = model.predict(X_train)
# absError_trainData = 0
# counter_trainData = 0
# correctPrediction_trainData = 0
# for i in range(len(predictions)):
#     if(training_data.target[i] <= 1.0):
#         error = abs(predictions[i] - training_data.target[i])
#         absError_trainData += error
#         counter_trainData +=1
#         if (error <= 0.01):
#             correctPrediction_trainData += 1
# print('abstrainingError = ', absError_trainData, 'correctPredictedTrainingData in percent =' ,correctPrediction_trainData/counter_trainData)

# predictions_val = model.predict(X_val)
# absError_valData = 0
# counter_valData = 0
# correctPrediction_valData = 0
# for i in range(len(predictions_val)):
#     if(Y_val[i] <= 1.0):
#         error = abs(predictions_val[i] - Y_val[i])
#         absError_valData += error
#         counter_valData +=1
#         if (error <= 0.01):
#             correctPrediction_valData += 1
# print('absDevError = ', absError_valData, 'correctPredictedDevData in percent =', correctPrediction_valData/counter_valData, "absNumberValSet=", counter_valData)

# class MyCustomCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         res_eval_1 = self.model.evaluate(X_test_1, y_test_1, verbose = 0)
#         res_eval_2 = self.model.evaluate(X_test_2, y_test_2, verbose = 0)
#         print(res_eval_1)
#         print(res_eval_2)

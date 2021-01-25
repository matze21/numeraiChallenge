import pandas as pd
import numerapi
import sklearn.linear_model

import keras_deepCNN
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

useValidationData=True

training_data = pd.read_csv("data/numerai_datasets_17.01.21/numerai_training_data.csv")
if useValidationData:
    validation_data = pd.read_csv("data/numerai_datasets_17.01.21/numerai_validation_data.csv")
    feature_cols_val = validation_data.columns[validation_data.columns.str.startswith('feature')]
    validation_features = validation_data[feature_cols_val]
    X_val = validation_features
    Y_val = validation_data.target
# find only the feature columns
feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
training_features = training_data[feature_cols]

model = keras_deepCNN.deepNN(training_features.shape[1])

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)#learning_rates_dec[0])
#model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])
#model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#define training data x and y
X_train = training_features
Y_train = training_data.target

history = model.fit(X_train, Y_train, epochs = 10, batch_size = 64, validation_data=(X_val, Y_val))
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
plt.show();

# model = sklearn.linear_model.LinearRegression(normalize = True)
# model.fit(training_features, training_data.target)
# predictions = model.predict(validation_features)
# predictions_df = validation_data["id"].to_frame()
# predictions_df["prediction_kazutsugi"] = predictions
# predictions_df.head()
predictions = model.predict(X_train)
absError_trainData = 0
counter_trainData = 0
correctPrediction_trainData = 0
for i in range(len(predictions)):
    if(training_data.target[i] <= 1.0):
        error = abs(predictions[i] - training_data.target[i])
        absError_trainData += error
        counter_trainData +=1
        if (error <= 0.01):
            correctPrediction_trainData += 1
print('abstrainingError = ', absError_trainData, 'correctPredictedTrainingData in percent =' ,correctPrediction_trainData/counter_trainData)

predictions_val = model.predict(X_val)
absError_valData = 0
counter_valData = 0
correctPrediction_valData = 0
for i in range(len(predictions_val)):
    if(Y_val[i] <= 1.0):
        error = abs(predictions_val[i] - Y_val[i])
        absError_valData += error
        counter_valData +=1
        if (error <= 0.01):
            correctPrediction_valData += 1
print('absDevError = ', absError_valData, 'correctPredictedDevData in percent =', correctPrediction_valData/counter_valData, "absNumberValSet=", counter_valData)

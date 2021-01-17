import pandas as pd
import numerapi
import sklearn.linear_model

import keras_deepCNN
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model

training_data = pd.read_csv("data/numerai_datasets_17.01.21/numerai_training_data.csv")

# find only the feature columns
feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
training_features = training_data[feature_cols]

model = keras_deepCNN.deepNN(training_features.shape[1])
model.compile(optimizer='adam', loss='logcosh', metrics=['accuracy'])

#define training data x and y
X_train = training_features
Y_train = training_data.target

model.fit(X_train, Y_train, epochs = 1, batch_size = 32)

model.save_weights("model.h5")

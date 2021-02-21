import pandas as pd
import numerapi
#import sklearn.linear_model

import keras_deepCNN
#from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

tournament_data = pd.read_csv("data/numerai_datasets_14.02.21/numerai_tournament_data.csv")
feature_cols_val = tournament_data.columns[tournament_data.columns.str.startswith('feature')]
validation_features = tournament_data[feature_cols_val]
X_val = validation_features

model = keras_deepCNN.deepNN(validation_features.shape[1])

#opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, decay=0e-8)#learning_rates_dec[0])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.load_weights("model_98perAcc.h5")

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

predictions_df = tournament_data["id"].to_frame()
predictions_df["prediction_kazutsugi"] = predictionVektor
predictions_df.head()

predictions_df.to_csv("predictions.csv", index=False)

# public_id  = "5PKOLW4ZJQDSMTC2QWPPG2QEHB427MFJ"
# secret_key = "I26YRNIBRQF47E6SO6VMLTLGN4O2MHL6ADUB4JNNQAYW3DPCH6JKY4HS5R2PYKLB"
# model_id   = "e994d440-764d-495d-8d60-7dbac3ac615b"
# napi = numerapi.NumerAPI(public_id=public_id, secret_key=secret_key)
# submission_id = napi.upload_predictions("predictions.csv", model_id=model_id)
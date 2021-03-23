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

useValidationData=True

tic = time.time()
training_data = pd.read_csv("data/numerai_datasets_21.02.21/numerai_training_data.csv")

feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
trainingData = trainingData_class0[feature_cols].reset_index().drop(['index'], axis = 1)


trainingData_class0   = training_data.loc[training_data.label_0 == 1]
trainingData_class025 = training_data.loc[training_data.label_025 == 1]
trainingData_class05  = training_data.loc[training_data.label_05 == 1]
trainingData_class075 = training_data.loc[training_data.label_075 == 1]
trainingData_class1   = training_data.loc[training_data.label_1 == 1]

train_X_label_0   = trainingData_class0[feature_cols].reset_index().drop(['index'], axis = 1)
train_X_label_025 = trainingData_class025[feature_cols].reset_index().drop(['index'], axis = 1)
train_X_label_05  = trainingData_class05[feature_cols].reset_index().drop(['index'], axis = 1)
train_X_label_075 = trainingData_class075[feature_cols].reset_index().drop(['index'], axis = 1)
train_X_label_1   = trainingData_class1[feature_cols].reset_index().drop(['index'], axis = 1)


if useValidationData:
    validation_data = pd.read_csv("data/numerai_datasets_21.02.21/numerai_tournament_data.csv")
    validationData_class0   = validation_data.loc[validation_data.target == 0]
    validationData_class025 = validation_data.loc[validation_data.target == 0.25]
    validationData_class05  = validation_data.loc[validation_data.target == 0.5]
    validationData_class075 = validation_data.loc[validation_data.target == 0.75]
    validationData_class1   = validation_data.loc[validation_data.target == 1.0]      
    X_label_0   = validationData_class0[feature_cols].reset_index().drop(['index'], axis = 1)
    X_label_025 = validationData_class025[feature_cols].reset_index().drop(['index'], axis = 1)
    X_label_05  = validationData_class05[feature_cols].reset_index().drop(['index'], axis = 1)
    X_label_075 = validationData_class075[feature_cols].reset_index().drop(['index'], axis = 1)
    X_label_1   = validationData_class1[feature_cols].reset_index().drop(['index'], axis = 1)
    Y_label_0   = validationData_class0.target.reset_index().drop(['index'], axis = 1)
    Y_label_025 = validationData_class025.target.reset_index().drop(['index'], axis = 1)
    Y_label_05  = validationData_class05.target.reset_index().drop(['index'], axis = 1)
    Y_label_075 = validationData_class075.target.reset_index().drop(['index'], axis = 1)
    Y_label_1   = validationData_class1.target.reset_index().drop(['index'], axis = 1)

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
        #Y_val = validation_data.target
# find only the feature columns



X_train = train_X_label_0


n_truePositives = X_train.shape[0]
n_classes =  int(n_truePositives / 3)

# Y_trainVec = Y_trainVec.append(train_Y_label_025.loc[0:n_classes], ignore_index = True)
# Y_trainVec = Y_trainVec.append(train_Y_label_05.loc[0:n_classes], ignore_index = True)
# Y_trainVec = Y_trainVec.append(train_Y_label_075.loc[0:n_classes], ignore_index = True)
# Y_trainVec = Y_trainVec.append(train_Y_label_1.loc[0:n_classes], ignore_index = True)       

X_train = X_train.append(train_X_label_025.loc[0:n_classes,:], ignore_index = True)
X_train = X_train.append(train_X_label_05.loc[0:n_classes,:], ignore_index = True)
X_train = X_train.append(train_X_label_075.loc[0:n_classes,:], ignore_index = True)
X_train = X_train.append(train_X_label_1.loc[0:n_classes,:], ignore_index = True)  

Y_train = np.zeros((X_train.shape[0], 1))
Y_train[0:n_truePositives] = 1


# X_train = training_data[feature_cols].to_numpy()
# Y_trainVec = training_data.label_0

# j=0
# Y_train_npArray = Y_trainVec.to_numpy()
# Y_train = np.zeros((Y_train_npArray.shape[0], 5))
# for j in range(Y_train_npArray.shape[0]):
#     if Y_train_npArray[j] == 0:
#         Y_train[j, 0] = 1
#     elif Y_train_npArray[j] == 0.25:
#         Y_train[j, 1] = 1
#     elif Y_train_npArray[j] == 0.5:
#         Y_train[j, 2] = 1
#     elif Y_train_npArray[j] == 0.75:
#         Y_train[j, 3] = 1
#     elif Y_train_npArray[j] == 1.0:
#         Y_train[j, 4] = 1
#     else:
#         print("something went wrong, new class", Y_train_npArray[j])


toc = time.time()
print("processed the data took ", toc - tic)

X_train, Y_train = shuffle(X_train, Y_train)

X_trainSet = X_train
Y_trainSet_0 = Y_train[:,0]

X_val_test = X_val
Y_val_test = Y_val[:,0]



model_0 = keras_deepCNN_singleClass.deepNN(X_train.shape[1])

#model_0.load_weights("model_singleClass.h5")
opt = Adam(lr=0.00005, beta_1=0.9, beta_2=0.9, decay=1e-8)
model_0.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model_0.fit(X_trainSet, Y_trainSet_0, epochs =30, batch_size = 256*128, validation_data=(X_val_test, Y_val_test))
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


predictionVektorTrainData = model_0.predict(X_trainSet)
dataFrame = pd.DataFrame(Y_trainSet_0, columns=['target'])
dataFrame['predictions'] = predictionVektorTrainData
dataFrame.to_csv('evaluate_model_train.csv')

predictionVektorDevData = model_0.predict(X_val_test)
dataFrame = pd.DataFrame(Y_val_test, columns=['target'])
dataFrame['predictions'] = predictionVektorDevData
dataFrame.to_csv('evaluate_model_test.csv')



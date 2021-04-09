import pandas as pd
#import numerapi
import sklearn.linear_model

# import keras_deepCNN
# from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import time

# gradient boosting for classification in scikit-learn
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from matplotlib import pyplot
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder 
from xgboost import XGBClassifier

useValidationData=True

tic = time.time()
# training_data = pd.read_csv("balancedTrainingData_test12021-01-31_2021-03-21.csv")

# feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
# X_train = training_data[feature_cols].to_numpy()

# #Y_train = training_data.loc[:,['label_0', 'label_025', 'label_05', 'label_075', 'label_1']].to_numpy()
# Y_train = training_data.loc[:,['target']].to_numpy()

#training_data = pd.read_csv("appendedTrainingData_test12021-01-31_2021-03-21.csv")
training_data = pd.read_csv("data/numerai_datasets_21.02.21/numerai_training_data.csv")

feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
X_train = training_data[feature_cols].reset_index().drop(['index'], axis = 1)


X_train_0 = X_train.to_numpy()
Y_train_0 = training_data.target
# Y_train_0 = Y_train_0.replace(0.5, 2)
# Y_train_0 = Y_train_0.replace([0.25, 0, 0.75, 1], 0)
# Y_train_0 = Y_train_0.replace(2, 1)
Y_train_0 = Y_train_0.to_numpy()

if useValidationData:
    tournament_data = pd.read_csv("data/numerai_datasets_21.02.21/numerai_tournament_data.csv")
    validation_data = tournament_data.loc[tournament_data.data_type == 'validation']
    X_val = validation_data[feature_cols].reset_index().drop(['index'], axis = 1).to_numpy()
    Y_val = validation_data.target
    Y_val_0 = Y_val
    X_val_0 = X_val
    # Y_val_0 = Y_val.replace(0.5, 2)
    # Y_val_0 = Y_val_0.replace([0.25, 0, 0.75, 1], 0)
    # Y_val_0 = Y_val_0.replace(2, 1)
    Y_val_0 = Y_val_0.to_numpy()

toc = time.time()
print("loaded the data took ", toc - tic)

# if useValidationData:
#     validation_data = pd.read_csv("data/numerai_datasets_21.03.21/numerai_tournament_data.csv")
#     trainingData_class0   = validation_data.loc[validation_data.target == 0]
#     trainingData_class025 = validation_data.loc[validation_data.target == 0.25]
#     trainingData_class05  = validation_data.loc[validation_data.target == 0.5]
#     trainingData_class075 = validation_data.loc[validation_data.target == 0.75]
#     trainingData_class1   = validation_data.loc[validation_data.target == 1.0]      
#     X_label_0   = trainingData_class0[feature_cols].reset_index().drop(['index'], axis = 1)
#     X_label_025 = trainingData_class025[feature_cols].reset_index().drop(['index'], axis = 1)
#     X_label_05  = trainingData_class05[feature_cols].reset_index().drop(['index'], axis = 1)
#     X_label_075 = trainingData_class075[feature_cols].reset_index().drop(['index'], axis = 1)
#     X_label_1   = trainingData_class1[feature_cols].reset_index().drop(['index'], axis = 1)
#     Y_label_0   = trainingData_class0.target.reset_index().drop(['index'], axis = 1)
#     Y_label_025 = trainingData_class025.target.reset_index().drop(['index'], axis = 1)
#     Y_label_05  = trainingData_class05.target.reset_index().drop(['index'], axis = 1)
#     Y_label_075 = trainingData_class075.target.reset_index().drop(['index'], axis = 1)
#     Y_label_1   = trainingData_class1.target.reset_index().drop(['index'], axis = 1)

#     #minLength = np.min([len(Y_label_0), len(Y_label_025), len(Y_label_05), len(Y_label_075), len(Y_label_1)])       
#     Y_val_data = Y_label_0#.loc[0:minLength]
#     Y_val_data = Y_val_data.append(Y_label_025, ignore_index = True)
#     Y_val_data = Y_val_data.append(Y_label_05, ignore_index = True)
#     Y_val_data = Y_val_data.append(Y_label_075, ignore_index = True)
#     Y_val_data = Y_val_data.append(Y_label_1, ignore_index = True)       
#     X_val = X_label_0
#     X_val = X_val.append(X_label_025, ignore_index = True)
#     X_val = X_val.append(X_label_05, ignore_index = True)
#     X_val = X_val.append(X_label_075, ignore_index = True)
#     X_val = X_val.append(X_label_1, ignore_index = True)  


#     j=0
#     Y_val_npArray = Y_val_data.to_numpy()
#     Y_val = Y_val_data
    # Y_val = np.zeros((Y_val_npArray.shape[0], 5))
    # for j in range(Y_val_npArray.shape[0]):
    #     if Y_val_npArray[j] == 0:
    #         Y_val[j, 0] = 1
    #     elif Y_val_npArray[j] == 0.25:
    #         Y_val[j, 1] = 1
    #     elif Y_val_npArray[j] == 0.5:
    #         Y_val[j, 2] = 1
    #     elif Y_val_npArray[j] == 0.75:
    #         Y_val[j, 3] = 1
    #     elif Y_val_npArray[j] == 1.0:
    #         Y_val[j, 4] = 1
    #     else:
    #         print("something went wrong, new class", Y_val_npArray[j])
        #Y_val = validation_data.target
# find only the feature columns



toc = time.time()
print("processed the data took ", toc - tic)


X_train, Y_train = shuffle(X_train_0, Y_train_0)

# le = LabelEncoder()
# Y_train = le.fit_transform(Y_train)
# Y_val = le.fit_transform(Y_val)

# evaluate the model
model = XGBClassifier(
        #base_score=0.0, booster='gbtree', colsample_bylevel=1,
    #    colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
    #    max_delta_step=0.0, max_depth=8, min_child_weight=1, missing=None,
        n_estimators=100000, #n_jobs=10, nthread=None,
        #objective='binary:logistic', random_state=0, reg_alpha=0.00,
    #    reg_lambda=1.0, scale_pos_weight=1, seed=None, silent=None,
    #    subsample=0.8, verbosity=0
       ) 
model.fit(X_train, Y_train)

#import pdb; pdb.set_trace()

#model.save_weights("model_gradientBoost.h5") 
Y_train_pred = model.predict(X_train)
#*** Test
train_acc = accuracy_score(Y_train, Y_train_pred)
print("train error = ", train_acc)


#validation score
y_pred = model.predict(X_val_0)

scores = accuracy_score(Y_val_0, y_pred)
print("crossvalidation score", scores)

cm = metrics.confusion_matrix(Y_val_0, y_pred)
print("confusion matrix", cm)
cr = metrics.classification_report(Y_val_0, y_pred)
print("classification report", cr)

from sklearn.externals import joblib

# Save to file in the current working directory
joblib_file = "joblib_model_gradientBoosting.pkl"
joblib.dump(model, joblib_file)

# # Load from file
# joblib_model = joblib.load(joblib_file)

# # Calculate the accuracy and predictions
# score = joblib_model.score(Xtest, Ytest)
# print("Test score: {0:.2f} %".format(100 * score))
# Ypredict = pickle_model.predict(Xtest)
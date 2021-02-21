import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from datetime import date, timedelta
import os

class makeDataBalanced():
    def __init__(self):
        self.start_date = date(2021,2, 21)
        self.delta = timedelta(days=7)
        
    def mergeTrainingDataDF(self):
        path = "data/numerai_datasets_"
        test_date = self.start_date
        data = pd.DataFrame()
        test_path = path + test_date.strftime("%d.%m.%y")
        while os.path.exists(test_path):
            balancedSet_trainingData = self.makeTrainingDataEnriched(test_path + "/numerai_training_data.csv")
            data = pd.concat([data, balancedSet_trainingData], axis = 0)
            if (test_date != date(2021,2, 21)):
                balancedSet_trainingData = self.makeTrainingDataEnriched(test_path + "/numerai_tournament_data.csv")
                data = pd.concat([data, balancedSet_trainingData], axis = 0)
            test_date = test_date + self.delta
            test_path = path + test_date.strftime("%d.%m.%y")
        data.to_csv('enrichedTrainingData_test1' + str(self.start_date) + '_' + str(test_date - self.delta) + '.csv')
        
    def makeTrainingDataEnriched(self, path_to_csv):
        training_data = pd.read_csv(str(path_to_csv))

        # find only the feature columns
        feature_cols = training_data.columns[training_data.columns.str.startswith('feature')]
        trainingData_class0   = training_data.loc[training_data.target == 0]
        trainingData_class025 = training_data.loc[training_data.target == 0.25]
        trainingData_class05  = training_data.loc[training_data.target == 0.5]
        trainingData_class075 = training_data.loc[training_data.target == 0.75]
        trainingData_class1   = training_data.loc[training_data.target == 1.0]      
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
        maxLength = np.max([len(Y_label_0), len(Y_label_025), len(Y_label_05), len(Y_label_075), len(Y_label_1)])


        import pdb; pdb.set_trace()
        n_label0 = maxLength%len(Y_label_0)
        rest_label0 = maxLength - n_label0 * len(Y_label_0)
        for i in range(n_label0):
            X_label_0 = pd.concat([X_label_0, X_label_0], axis = 0) 
            Y_label_0 = pd.concat([Y_label_0, Y_label_0], axis = 0) 

        n_label0 = maxLength%len(Y_label_025)
        rest_label0 = maxLength - n_label0 * len(Y_label_0)
        for i in range(n_label0):
            X_label_025 = pd.concat([X_label_025, X_label_025], axis = 0) 
            Y_label_025 = pd.concat([Y_label_025, Y_label_025], axis = 0) 
        
        n_label0 = maxLength%len(Y_label_05)
        rest_label0 = maxLength - n_label0 * len(Y_label_0)
        for i in range(n_label0):
            X_label_05 = pd.concat([X_label_05, X_label_05], axis = 0) 
            Y_label_05 = pd.concat([Y_label_05, Y_label_05], axis = 0) 
        
        n_label0 = maxLength%len(Y_label_075)
        rest_label0 = maxLength - n_label0 * len(Y_label_0)
        for i in range(n_label0):
            X_label_075 = pd.concat([X_label_075, X_label_075], axis = 0) 
            Y_label_075 = pd.concat([Y_label_075, Y_label_075], axis = 0) 
        
        n_label0 = maxLength%len(Y_label_1)
        rest_label0 = maxLength - n_label0 * len(Y_label_0)
        for i in range(n_label0):
            X_label_1 = pd.concat([X_label_1, X_label_1], axis = 0) 
            Y_label_1 = pd.concat([Y_label_1, Y_label_1], axis = 0) 


        Y_train = Y_label_0
        Y_train = Y_train.append(Y_label_025, ignore_index = True)
        Y_train = Y_train.append(Y_label_05, ignore_index = True)
        Y_train = Y_train.append(Y_label_075, ignore_index = True)
        Y_train = Y_train.append(Y_label_1, ignore_index = True)       
        X_train = X_label_0
        X_train = X_train.append(X_label_025, ignore_index = True)
        X_train = X_train.append(X_label_05, ignore_index = True)
        X_train = X_train.append(X_label_075, ignore_index = True)
        X_train = X_train.append(X_label_1, ignore_index = True)       
        X_train_uniform = X_train.to_numpy()
        Y_train_uniform = Y_train.to_numpy()        
        #import pdb; pdb.set_trace()        
        X_train_uniform, Y_train_uniform = shuffle(X_train_uniform, Y_train_uniform)        
        X_train = X_train_uniform
        #Y_train = Y_train_uniform      
        Y_train = np.zeros((Y_train_uniform.shape[0], 5))
        #import pdb; pdb.set_trace()
        i=0
        for i in range(Y_train_uniform.shape[0]):
            if Y_train_uniform[i] == 0:
                Y_train[i, 0] = 1
            elif Y_train_uniform[i] == 0.25:
                Y_train[i, 1] = 1
            elif Y_train_uniform[i] == 0.5:
                Y_train[i, 2] = 1
            elif Y_train_uniform[i] == 0.75:
                Y_train[i, 3] = 1
            elif Y_train_uniform[i] == 1.0:
                Y_train[i, 4] = 1
            else:
                print("something went wrong, new class", Y_train_uniform[i])

        dataFrame = pd.DataFrame(X_train, columns=feature_cols)

        Y_trainDF = pd.DataFrame(Y_train, columns = ['label_0', 'label_025', 'label_05', 'label_075', 'label_1'])
        dataFrame = pd.concat([dataFrame, Y_trainDF], axis = 1)

        return dataFrame

dataProcessing = makeDataBalanced()
dataProcessing.mergeTrainingDataDF()

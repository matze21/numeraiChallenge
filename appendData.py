import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from datetime import date, timedelta
import os

class makeDataBalanced():
    def __init__(self):
        self.start_date = date(2021,1, 31)
        self.delta = timedelta(days=7)
        
    def mergeTrainingDataDF(self):
        path = "data/numerai_datasets_"
        test_date = self.start_date
        data = pd.DataFrame()
        test_path = path + test_date.strftime("%d.%m.%y")
        while os.path.exists(test_path):
            balancedSet_trainingData = self.makeTrainingDataBalanced(test_path + "/numerai_training_data.csv")
            data = pd.concat([data, balancedSet_trainingData], axis = 0)
            test_date = test_date + self.delta
            test_path = path + test_date.strftime("%d.%m.%y")
        data.to_csv('appendedTrainingData_test1' + str(self.start_date) + '_' + str(test_date - self.delta) + '.csv')
        
    def makeTrainingDataBalanced(self, path_to_csv):
        training_data = pd.read_csv(str(path_to_csv))

        # find only the feature columns
        feature_cols = training_data.columns[training_data.columns.str.startswith('feature')] 
        X_train   = training_data[feature_cols]
        Y_train   = training_data.target

        X_train_uniform = X_train.to_numpy()
        Y_train_uniform = Y_train.to_numpy()        
        #import pdb; pdb.set_trace()        
        X_train_uniform, Y_train_uniform = shuffle(X_train_uniform, Y_train_uniform)        
        X_train = X_train_uniform
        Y_train = Y_train_uniform      
        #Y_train = np.zeros((Y_train_uniform.shape[0], 5))
        #import pdb; pdb.set_trace()
        # i=0
        # for i in range(Y_train_uniform.shape[0]):
        #     if Y_train_uniform[i] == 0:
        #         Y_train[i, 0] = 1
        #     elif Y_train_uniform[i] == 0.25:
        #         Y_train[i, 1] = 1
        #     elif Y_train_uniform[i] == 0.5:
        #         Y_train[i, 2] = 1
        #     elif Y_train_uniform[i] == 0.75:
        #         Y_train[i, 3] = 1
        #     elif Y_train_uniform[i] == 1.0:
        #         Y_train[i, 4] = 1
        #     else:
        #         print("something went wrong, new class", Y_train_uniform[i])

        dataFrame = pd.DataFrame(X_train, columns=feature_cols)

        # Y_trainDF = pd.DataFrame(Y_train, columns = ['label_0', 'label_025', 'label_05', 'label_075', 'label_1'])
        # dataFrame = pd.concat([dataFrame, Y_trainDF], axis = 1)
        dataFrame['target'] = Y_train_uniform

        return dataFrame

dataProcessing = makeDataBalanced()
dataProcessing.mergeTrainingDataDF()

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from datetime import date, timedelta
import os

class splitData():
    def __init__(self):
        self.start_date = date(2021,1, 31)
        self.delta = timedelta(days=7)

    def splitDataIfNotAlreadyDone(self):
        path = "data/numerai_datasets_"
        test_date = self.start_date
        data = pd.DataFrame()
        test_path = path + test_date.strftime("%d.%m.%y")
        allowedWeeksWithoutData = 5
        weeksWithoutData = allowedWeeksWithoutData
        while os.path.exists(test_path) or weeksWithoutData > 0:
            path_to_validation = test_path + "/numerai_validation_data.csv"
            path_to_tournament_data = test_path +  "/numerai_tournament_data.csv"
            print("check for ", test_path)
            #only split data if not already exist
            if not os.path.exists(path_to_validation) and os.path.exists(path_to_tournament_data):
                print("found tournament and splitting folder: ", test_path)
                
                tournament_data = pd.read_csv(str(path_to_tournament_data))
                feature_cols = tournament_data.columns[tournament_data.columns.str.startswith('feature')] 

                validation_data = tournament_data.loc[tournament_data.data_type == 'validation']
                # validation_data[feature_cols] = validation_data[feature_cols].astype(np.float16)
                # validation_data.target        = validation_data.target.astype(np.float16)

                validation_data.to_csv(path_to_validation)
                print("success..")
            else:
                weeksWithoutData -= 1
                print("no tournament data found!") 
            test_date = test_date + self.delta
            test_path = path + test_date.strftime("%d.%m.%y")

dataProcessing = splitData()
dataProcessing.splitDataIfNotAlreadyDone()
import pandas as pd
from src import config
from sklearn.preprocessing import LabelEncoder
import joblib

def feature_process(data,mode):
    df = pd.read_csv(data)

    # Feature imputation
    df.society.fillna("No society",inplace=True)
    df.bhk.fillna(0,inplace=True)
    df.bath.fillna(1,inplace=True)
    df.balcony.fillna(0,inplace=True)

    # label encoding categorical variables
    labelencoders = {}

    for c in ["area_type","availability","location","society"]:
        labelencoders[c] = LabelEncoder()
        if mode=='train':
            df[c] = labelencoders[c].fit_transform(df[c])
        else:
            labelencoders = joblib.load(config.MODELS_PATH+"feature_encoders.pkl")
            df[c] = labelencoders[c].transform(df[c])



    df.to_csv(config.TRAIN_PROCESSED_DATA,index=False)

    if mode == 'train':
        joblib.dump(labelencoders,config.MODELS_PATH+'feature_encoders.pkl')

feature_process(config.TRAIN_DATA,'train')
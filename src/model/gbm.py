import xgboost as xgb
from src.model.ml_common.Loss import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import pandas as pd
import pickle

"""
xgb API: https://xgboost.readthedocs.io/en/latest/python/python_api.html
"""
class XgbModel(object):

    def __init__(self, params=None, isCustomLoss=False):
        self.params       = params
        self.isCustomLoss = isCustomLoss
        self.trainDf      = None
        self.testDf       = None
        self.dataPath     = "../data/DenseFeature.csv"


    def load_data(self):
        data = pd.read_csv(self.dataPath)
        self.trainDf = data.sample(frac=0.9, replace=False, random_state=0, axis=0)
        self.testDf  = data[~data.index.isin(self.trainDf.index)]


    def train(self):
        columns = self.trainDf.columns
        feature_cols, label_col = columns[1:], columns[:1]
        X_train, X_valid, y_train, y_valid = train_test_split(self.trainDf[feature_cols], self.trainDf[label_col], test_size=0.1, random_state=12345)

        dtrain, dvalid = xgb.DMatrix(X_train, y_train), xgb.DMatrix(X_valid, y_valid)
        watchlist      = [(dvalid, 'eval'), (dtrain, 'train')]
        if self.isCustomLoss:
            model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=50, evals=watchlist, obj=xgbCateLoss, early_stopping_rounds=5)
        else:
            model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=50, evals=watchlist, early_stopping_rounds=5)

        pickle.dump(model, open("../modelSave/xgboost.pickle.dat", "wb"))

    def predict(self):
        model = pickle.load(open("../modelSave/xgboost.pickle.dat", "rb"))
        columns = self.testDf.columns
        feature_cols, label_col = columns[1:], columns[:1]
        X_test, y_test = self.testDf[feature_cols], self.testDf[label_col]
        dtest    = xgb.DMatrix(X_test)
        y_pred   = model.predict(dtest)
        accuracy = average_precision_score(y_test, y_pred)
        print("accuracy: %.2f%%" % (accuracy*100.0))
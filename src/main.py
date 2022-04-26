from data.script.ConstructData import *
from src.model.tf_common.utils import *
import tensorflow as tf
from src.model.tf_common.inputs import LoadData
from src.model.lr import LRModel
from src.model.fm import FMModel
from src.model.gbm import XgbModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def productData(dense=12, sparse=6, varlen=2, sample=10000):
    constructData = ConstructData(dense, sparse, varlen, sample)
    constructData.generateData()

def xgb():
    params = {
        "booster": "gbtree",
        "eta": 0.1,
        "max_depth": 4,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "nthread": 3,
        "seed": 0,
        "silent": 0
    }

    xgb = XgbModel(params, False)
    xgb.load_data()
    xgb.train()
    xgb.predict()

def lr():
    productData(dense=12, sparse=0, varlen=0, sample=10000)
    loadData     = LoadData(col_columns=LR_col_columns, feature_columns=LR_feature_columns, DEFAULT_VALUES=LR_DEFAULT_VALUES, batchSize=256)
    train, valid = loadData.load_data()

    model        = LRModel(LR_feature_columns)
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.metrics.binary_crossentropy, metrics=['accuracy'])
    model.fit(train, validation_data=valid, epochs=5)

def fm():
    productData(dense=12, sparse=6, varlen=2, sample=10000)
    loadData     = LoadData(col_columns=FM_col_columns, feature_columns=FM_feature_columns, DEFAULT_VALUES=FM_DEFAULT_VALUES, batchSize=256)
    train, valid = loadData.load_data()

    model        = FMModel(FM_feature_columns)
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.metrics.binary_crossentropy, metrics=['accuracy'])
    model.fit(train, validation_data=valid, epochs=3, workers=4)

def main():
    lr()
    xgb()
    fm()

if __name__ == '__main__':
    main()
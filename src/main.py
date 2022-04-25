from data.script.ConstructData import *
from src.model.tf_common.utils import *
import tensorflow as tf
from src.model.tf_common.inputs import load_data
from src.model.lr import LRModel
from src.model.fm import FMModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from src.model.gbm import XgbModel

dense_columns  = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11']
sparse_columns = ['d0', 'd1', 'd2', 'd3', 'd4', 'd5']
varlen_columns = ['m0', 'm1']

def productData():
    constructData = ConstructData(12, 6, 2, 1000000)
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
    denseFeatures  = pd.read_csv("../data/DenseFeature.csv")
    sparseFeatures = pd.read_csv("../data/SparseFeature.csv")
    sparseFeatures = sparseFeatures[sparse_columns]
    data           = pd.concat([denseFeatures, sparseFeatures], axis=1)
    train_X        = [data[name].values for name in sparse_columns+dense_columns]
    train_Y        = data['label'].values

    feature_columns = []
    for col in dense_columns:
        denseFeat      = DenseFeat(name=col, dtype=tf.float32)
        feature_columns.append(denseFeat)

    for col in sparse_columns:
        sparseFeat     = SparseFeat(name=col, dtype=tf.float32)
        feature_columns.append(sparseFeat)

    model = LRModel(feature_columns)
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.metrics.binary_crossentropy, metrics=['accuracy'])

    model.fit(train_X, train_Y, batch_size=32, epochs=5)

def fm():
    train, valid = load_data(batch_size=1024)
    model        = FMModel(feature_columns)
    model.compile(tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.metrics.binary_crossentropy, metrics=['accuracy'])
    model.fit(train, validation_data=valid, epochs=3, workers=4)

def main():
    productData()
    xgb()
    lr()
    fm()

if __name__ == '__main__':
    main()
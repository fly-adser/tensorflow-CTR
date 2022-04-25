import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from src.model.tf_common.feature_columns import build_input_features

"""
docs link: https://zhuanlan.zhihu.com/p/56900935
"""
def LRModel(feature_columns):
    """
    param feature_columns: An iterable containing all the features used by LR model
    """
    features   = build_input_features(feature_columns)
    input_list = list(features.values())
    x          = Concatenate(axis=1)(input_list)
    outputs    = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=True)(x)
    model      = tf.keras.Model(inputs=input_list, outputs=outputs)

    return model
import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from src.model.tf_commonV2.interaction import FMLayer
from src.model.tf_commonV2.feature_columns import build_input_features, input_from_feature_columns

"""
paper link: https://arxiv.org/abs/1703.04247
"""
def DeepFMModel(feature_columns):
    """
    param feature_columns: An iterable containing all the features used by DeepFM model
    """
    features   = build_input_features(feature_columns)
    input_list = features.values()

    sparse_embedding_list, _ = input_from_feature_columns(features, feature_columns, l2_reg=0.001, prefix='deepfm')
    sparse_embedding_kd      = Concatenate(axis=1)(sparse_embedding_list)

    fm_logit = FMLayer()(sparse_embedding_kd)

    fc0 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)(sparse_embedding_kd)
    fc1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(fc0)
    fc2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)(fc2)
    fc4 = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(fc3)
    nn_logit = tf.keras.layers.Flatten()(fc4)

    final_logit = Concatenate(axis=1)([fm_logit, nn_logit])
    outputs     = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=True)(final_logit)
    model       = tf.keras.Model(inputs=input_list, outputs=outputs)

    return model
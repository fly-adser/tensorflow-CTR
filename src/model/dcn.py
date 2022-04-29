import tensorflow as tf
from src.model.tf_commonV2.feature_columns import build_input_features, input_from_feature_columns

"""
paper link: https://arxiv.org/abs/1708.05123
"""

def DCNModel(feature_columns):
    """
    param feature_columns: An iterable containing all the features used by DeepFM model
    """
    features   = build_input_features(feature_columns)
    input_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns, l2_reg=0.001, prefix='deepfm')
    inputs = tf.keras.layers.Concatenate(axis=1)(sparse_embedding_list+dense_value_list)
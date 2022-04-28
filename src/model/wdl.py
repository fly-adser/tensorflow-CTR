import tensorflow as tf
from tensorflow.keras.layers import Concatenate
from src.model.tf_commonV2.feature_columns import build_input_features, get_linear_logit, input_from_feature_columns

"""
paper link: https://arxiv.org/pdf/1606.07792.pdf
"""
def wdl(linear_feature_columns, nn_feature_columns):
    """
    linear_feature_columns: An iterable containing all the features used by wide part
    nn_feature_columns: An iterable containing all the features used by deep part
    """
    features   = build_input_features(linear_feature_columns+nn_feature_columns)
    input_list = list(features.values())

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, linear_feature_columns, l2_reg=0.001, prefix='linear')
    linear_logit = get_linear_logit(sparse_embedding_list, dense_value_list)

    x0      = Concatenate(axis=1)(linear_logit)
    outputs = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=True)(x0)
    model   = tf.keras.Model(inputs=input_list, outputs=outputs)

    return model
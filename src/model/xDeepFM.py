import tensorflow as tf
from src.model.tf_commonV2.interaction import CIN
from src.model.tf_commonV2.feature_columns import build_input_features, get_linear_logit, input_from_feature_columns, flatten

"""
paper link: https://arxiv.org/abs/1803.05170
"""
def xDeepFMModel(feature_columns):
    """
    param feature_columns: An iterable containing all the features used by xDeepFM model
    """
    features   = build_input_features(feature_columns)
    input_list = features.values()

    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns, l2_reg=0.001, prefix='xdeepfm')

    linear_logit = get_linear_logit([], dense_value_list)

    cin_input    = tf.keras.layers.Concatenate(axis=1)(sparse_embedding_list)
    cin_logit    = CIN(activation='relu', split_half=False, seed=0)(cin_input)

    nn_input     = tf.keras.layers.Concatenate(axis=1)(flatten(sparse_embedding_list))
    fc0 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)(nn_input)
    fc1 = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)(fc0)
    fc2 = tf.keras.layers.Dense(units=64, activation=tf.nn.relu)(fc1)
    fc3 = tf.keras.layers.Dense(units=32, activation=tf.nn.relu)(fc2)
    fc4 = tf.keras.layers.Dense(units=1, activation=tf.nn.relu)(fc3)
    nn_logit = tf.keras.layers.Flatten()(fc4)

    final_logit = tf.keras.layers.Concatenate(axis=1)([linear_logit, cin_logit, nn_logit])
    outputs = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=True)(final_logit)
    model = tf.keras.Model(inputs=input_list, outputs=outputs)

    return model
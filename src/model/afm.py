import tensorflow as tf
from src.model.tf_commonV2.feature_columns import build_input_features, get_linear_logit, input_from_feature_columns
from src.model.tf_commonV2.interaction import AFMLayer

"""
paper link: https://arxiv.org/abs/1708.04617
"""
def AFMModel(feature_columns):
    """
    param feature_columns: An iterable containing all the features used by AFM model
    """
    features   = build_input_features(feature_columns)
    input_list = features.values()

    sparse_enbedding_list, dense_value_list = input_from_feature_columns(features, feature_columns, l2_reg=0.01, linear_dim=1, prefix='linear')
    linear_logit = get_linear_logit(sparse_enbedding_list, dense_value_list)

    sparse_enbedding_list, _ = input_from_feature_columns(features, feature_columns, l2_reg=0.01, prefix='afm')
    afm_logit    = AFMLayer(attention_factor=4, l2_reg_w=0.01, dropout_rate=0, seed=0)(sparse_enbedding_list)

    final_logit = tf.keras.layers.Concatenate(axis=1)([linear_logit, afm_logit])
    outputs = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=True)(final_logit)
    model = tf.keras.Model(inputs=input_list, outputs=outputs)

    return model
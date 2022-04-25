from tensorflow.keras import Model
from src.model.tf_common.feature_columns import *
from src.model.tf_common.layers import *
from src.model.tf_common.utils import *

"""
docs link: https://zhuanlan.zhihu.com/p/397166601
"""
def FMModel(feature_columns):
    """
    param feature_columns: An iterable containing all the features used by LR model
    """
    features   = build_input_features(feature_columns)
    input_list = features.values()

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    sparse_varlen_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    linear_embedding_dict  = build_linear_embedding_dict(feature_columns)
    linear_sparse_embedding_list, dense_value_list = input_from_feature_columns(features, feature_columns, linear_embedding_dict)
    linear_logit           = get_linear_logit(linear_sparse_embedding_list, dense_value_list)

    cross_columns          = sparse_feature_columns + sparse_varlen_feature_columns
    embedding_dict         = build_embedding_dict(cross_columns)
    sparse_embedding_dict, _  = input_from_feature_columns(features, cross_columns, embedding_dict)
    concat_sparse_kd_embed = Concatenate(axis=1)(sparse_embedding_dict)
    fm_cross_logit         = FMLayer()(concat_sparse_kd_embed)

    fm_logit = Add()([fm_cross_logit, linear_logit])
    output   = tf.keras.layers.Activation("sigmoid", name="fm_out")(fm_logit)
    model = Model(inputs=input_list, outputs=output)

    return model
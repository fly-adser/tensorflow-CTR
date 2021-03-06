from copy import copy

from tensorflow.keras import Input
import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras.layers import Concatenate, Flatten, Dense
from tensorflow.keras.initializers import Zeros
from src.model.tf_commonV2.utils import DenseFeat, SparseFeat, VarLenSparseFeat
from src.model.tf_commonV2.inputs import create_embedding_matrix, embedding_lookup, get_dense_input, get_varlen_pooling_list
from src.model.tf_commonV2.layers import Add

def build_input_features(features_columns, prefix=''):
    input_features = OrderedDict()

    for feat_col in features_columns:
        if isinstance(feat_col, DenseFeat):
            input_features[feat_col.name] = Input(shape=(1,), name=prefix+feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, SparseFeat):
            input_features[feat_col.name] = Input((1,), name=prefix+feat_col.name, dtype=feat_col.dtype)
        elif isinstance(feat_col, VarLenSparseFeat):
            input_features[feat_col.name] = Input((feat_col.maxlen,), name=prefix+feat_col.name, dtype=feat_col.dtype)
            if feat_col.weight_name is not None:
                input_features[feat_col.weight_name] = Input((feat_col.maxlen, ), name=prefix+feat_col.weight_name, dtype=tf.float32)
        else:
            raise TypeError("Invalid feature column in build_input_features: {}".format(feat_col.name))

    return input_features

def input_from_feature_columns(features, feature_columns, l2_reg, linear_dim=None, prefix=''):
    linear_feature_columns = []
    if linear_dim:
        linear_feature_columns = copy(feature_columns)
        for i in range(len(linear_feature_columns)):
            if isinstance(linear_feature_columns[i], SparseFeat):
                linear_feature_columns[i] = linear_feature_columns[i]._replace(embed_dim=1, embed_initializer=Zeros())
            if isinstance(linear_feature_columns[i], VarLenSparseFeat):
                linear_feature_columns[i] = linear_feature_columns[i]._replace(embed_dim=1, embed_initializer=Zeros())
    use_columns = linear_feature_columns if linear_dim else feature_columns

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), use_columns)) if use_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), use_columns)) if use_columns else []

    embedding_matrix_dict = create_embedding_matrix(use_columns, l2_reg, prefix)
    sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    varlen_embedding_dict = embedding_lookup(embedding_matrix_dict, features, varlen_sparse_feature_columns)
    dense_value_list      = get_dense_input(features, use_columns)
    varlen_polling_embed  = get_varlen_pooling_list(varlen_embedding_dict, features, varlen_sparse_feature_columns)

    return list(sparse_embedding_dict.values())+list(varlen_polling_embed.values()), dense_value_list

def concat_func(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return Concatenate(axis=axis)(inputs)

def get_linear_logit(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        linear_logit = concat_func([dense_linear_layer, sparse_linear_layer])
        return linear_logit
    elif len(sparse_embedding_list) > 0:
        sparse_linear_layer = Add()(sparse_embedding_list)
        sparse_linear_layer = Flatten()(sparse_linear_layer)
        return sparse_linear_layer
    elif len(dense_value_list) > 0:
        dense_linear = concat_func(dense_value_list)
        dense_linear_layer = Dense(1)(dense_linear)
        return dense_linear_layer
    else:
        raise Exception("linear_feature_columns can not be empty list")

def flatten(sparse_embedding_list):
    embedding_list = []
    for embedding in sparse_embedding_list:
        embedding_list.append(tf.keras.layers.Flatten()(embedding))

    return embedding_list

def reduce_sum(input_tensor,
               axis=None,
               keep_dims=False,
               name=None,
               reduction_indices=None):
    try:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keep_dims=keep_dims,
                             name=name,
                             reduction_indices=reduction_indices)
    except TypeError:
        return tf.reduce_sum(input_tensor,
                             axis=axis,
                             keepdims=keep_dims,
                             name=name)

def softmax(logits, dim=-1, name=None):
    try:
        return tf.nn.softmax(logits, dim=dim, name=name)
    except TypeError:
        return tf.nn.softmax(logits, axis=dim, name=name)
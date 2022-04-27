from tensorflow.keras.layers import Embedding
from tensorflow.keras.regularizers import l2
from collections import defaultdict
from src.model.tf_commonV2.utils import *
from src.model.tf_commonV2.layers import HashLayer, VocabLayer, WeightedSequenceLayer, SequencePollingLayer

def create_embedding_dict(feature_columns, l2_reg, prefix='sparse'):
    sparse_embedding = {}
    for feat in feature_columns:
        name = feat.share_embed if feat.share_embed else feat.name
        emb  = Embedding(feat.voc_size, feat.embed_dim, embeddings_initializer=feat.embed_initializer,
                         embeddings_regularizer=l2(l2_reg), name=prefix+'_emb_'+feat.name)
        sparse_embedding[name] = emb

    return sparse_embedding

def create_embedding_matrix(feature_columns, l2_reg, prefix=""):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    sparse_emb_dict = create_embedding_dict(sparse_feature_columns+varlen_feature_columns, l2_reg, prefix+'sparse')
    return sparse_emb_dict

def embedding_lookup(sparse_embedding_matrix, sparse_input_dict, sparse_feature_columns):
    sparse_embedding_dict = defaultdict(list)
    for feat in sparse_feature_columns:
        name = feat.share_embed if feat.share_embed else feat.name
        lookup_idx = sparse_input_dict[name]
        if feat.use_hash:
            lookup_idx = HashLayer(num_buckets=feat.voc_size)(lookup_idx)
        else:
            keys       = DICT_CATEGORICAL[name]
            lookup_idx = VocabLayer(keys)(lookup_idx)

        sparse_embedding_dict[name] = sparse_embedding_matrix[name](lookup_idx)

    return sparse_embedding_dict

def get_dense_input(features, feature_columns):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc.name])

    return dense_input_list

def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns):
    embedding_polling_list = defaultdict(list)
    for feat in varlen_sparse_feature_columns:
        voc_name = feat.share_embed if feat.share_embed else feat.name
        if feat.weight_name:
            seq_input = WeightedSequenceLayer()([embedding_dict][voc_name], features[feat.weight_name])
        else:
            seq_input = embedding_dict[voc_name]
        embedding_polling = SequencePollingLayer(feat.combiner)(seq_input)
        embedding_polling_list[voc_name] = embedding_polling

    return embedding_polling_list
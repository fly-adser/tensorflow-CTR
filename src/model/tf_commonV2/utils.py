from collections import namedtuple

class DenseFeat(namedtuple('DenseFeat', ['name', 'dtype'])):

    def __new__(cls, name, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dtype)

class SparseFeat(namedtuple('SparseFeat', ['name', 'voc_size', 'use_hash', 'embed_dim', 'share_embed', 'embed_initializer', 'dtype'])):

    def __new__(cls, name, voc_size, use_hash=False, embed_dim=4,  share_embed=None, embed_initializer=None, dtype='int32'):
        return super(SparseFeat, cls).__new__(name, voc_size, use_hash, embed_dim, share_embed, embed_initializer, dtype)

class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'use_hash', 'embed_dim', 'share_embed', 'combiner',
                                                       'embed_initializer', 'maxLen' ,'dtype'])):

    def __new__(cls, name, voc_size, use_hash=False, embed_dim=4,  share_embed=None, combiner='mean', embed_initializer=None, maxLen=3, dtype='int32'):
        return super(SparseFeat, cls).__new__(name, voc_size, use_hash, embed_dim, share_embed, combiner, embed_initializer, maxLen, dtype)

DICT_CATEGORICAL = {
    'd0': [i for i in range(1, 11)],
    'd1': [i for i in range(1, 11)],
    'd2': [i for i in range(1, 11)],
    'd3': [i for i in range(1, 11)],
    'd4': [i for i in range(1, 11)],
    'd5': [i for i in range(1, 11)]
}
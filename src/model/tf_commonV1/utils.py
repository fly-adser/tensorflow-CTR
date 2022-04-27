from collections import namedtuple
import tensorflow as tf

class DenseFeat(namedtuple('DenseFeat', ['name', 'dtype'])):

    def __new__(cls, name, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dtype)

class SparseFeat(namedtuple('SparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed', 'embed_dim', 'dtype'])):

    def __new__(cls, name, voc_size=1, hash_size=None, share_embed=None, embed_dim=4, dtype="int32"):
        return super(SparseFeat, cls).__new__(cls, name, voc_size, hash_size, share_embed, embed_dim, dtype)

class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'hash_size', 'share_embed', 'weight_name', 'combiner', 'embed_dim','maxlen', 'dtype'])):

    def __new__(cls, name, voc_size=1, hash_size=None, share_embed=None, weight_name=None, combiner=None, embed_dim=4, maxlen=3, dtype="int32"):
        return super(VarLenSparseFeat, cls).__new__(cls, name, voc_size, hash_size, share_embed, weight_name, combiner, embed_dim, maxlen, dtype)

DICT_CATEGORICAL = {
    'd0': [i for i in range(1, 11)],
    'd1': [i for i in range(1, 11)],
    'd2': [i for i in range(1, 11)],
    'd3': [i for i in range(1, 11)],
    'd4': [i for i in range(1, 11)],
    'd5': [i for i in range(1, 11)]
}

LR_DEFAULT_VALUES = [
    [0],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]
]

LR_col_columns = [
    'label',
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11'
]

LR_feature_columns = [
    DenseFeat(name='s0', dtype=tf.float32),
    DenseFeat(name='s1', dtype=tf.float32),
    DenseFeat(name='s2', dtype=tf.float32),
    DenseFeat(name='s3', dtype=tf.float32),
    DenseFeat(name='s4', dtype=tf.float32),
    DenseFeat(name='s5', dtype=tf.float32),
    DenseFeat(name='s6', dtype=tf.float32),
    DenseFeat(name='s7', dtype=tf.float32),
    DenseFeat(name='s8', dtype=tf.float32),
    DenseFeat(name='s9', dtype=tf.float32),
    DenseFeat(name='s10', dtype=tf.float32),
    DenseFeat(name='s11', dtype=tf.float32)
]

FM_DEFAULT_VALUES = [
    [0],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
    ['0:1.0;0:1.0;0:1.0'], ['0:1.0;0:1.0;0:1.0']
]

FM_col_columns = [
    'label',
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
    'd0', 'd1', 'd2', 'd3', 'd4', 'd5',
    'm0', 'm1'
]

FM_feature_columns = [
    SparseFeat(name='d0', voc_size=12, hash_size=12, share_embed=None, embed_dim=4, dtype=tf.string),
    SparseFeat(name='d1', voc_size=12, hash_size=12, share_embed=None, embed_dim=4, dtype=tf.string),
    SparseFeat(name='d2', voc_size=12, hash_size=12, share_embed=None, embed_dim=4, dtype=tf.string),
    SparseFeat(name='d3', voc_size=12, hash_size=12, share_embed='d0', embed_dim=4, dtype=tf.string),
    SparseFeat(name='d4', voc_size=12, hash_size=12, share_embed='d1', embed_dim=4, dtype=tf.string),
    SparseFeat(name='d5', voc_size=12, hash_size=12, share_embed='d2', embed_dim=4, dtype=tf.string),

    VarLenSparseFeat(name='m0', voc_size=6, hash_size=6, weight_name='m0_weight', combiner='mean', embed_dim=4, maxlen=3, dtype=tf.string),
    VarLenSparseFeat(name='m1', voc_size=6, hash_size=6, weight_name='m1_weight', combiner='mean', embed_dim=4, maxlen=3, dtype=tf.string),

    DenseFeat(name='s0', dtype=tf.float32),
    DenseFeat(name='s1', dtype=tf.float32),
    DenseFeat(name='s2', dtype=tf.float32),
    DenseFeat(name='s3', dtype=tf.float32),
    DenseFeat(name='s4', dtype=tf.float32),
    DenseFeat(name='s5', dtype=tf.float32),
    DenseFeat(name='s6', dtype=tf.float32),
    DenseFeat(name='s7', dtype=tf.float32),
    DenseFeat(name='s8', dtype=tf.float32),
    DenseFeat(name='s9', dtype=tf.float32),
    DenseFeat(name='s10', dtype=tf.float32),
    DenseFeat(name='s11', dtype=tf.float32)
]

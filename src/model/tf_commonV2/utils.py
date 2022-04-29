from collections import namedtuple
import tensorflow as tf

class DenseFeat(namedtuple('DenseFeat', ['name', 'dtype'])):

    def __new__(cls, name, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dtype)

class SparseFeat(namedtuple('SparseFeat', ['name', 'voc_size', 'use_hash', 'embed_dim', 'share_embed', 'embed_initializer', 'dtype'])):

    def __new__(cls, name, voc_size, use_hash=False, embed_dim=4,  share_embed=None, embed_initializer=None, dtype='int32'):
        return super(SparseFeat, cls).__new__(cls, name, voc_size, use_hash, embed_dim, share_embed, embed_initializer, dtype)

class VarLenSparseFeat(namedtuple('VarLenSparseFeat', ['name', 'voc_size', 'use_hash', 'weight_name', 'embed_dim', 'share_embed', 'combiner',
                                                       'embed_initializer', 'maxlen' ,'dtype'])):

    def __new__(cls, name, voc_size, use_hash=False, weight_name=None, embed_dim=4,  share_embed=None, combiner='mean', embed_initializer=None, maxlen=3, dtype='int32'):
        return super(VarLenSparseFeat, cls).__new__(cls, name, voc_size, use_hash, weight_name, embed_dim, share_embed, combiner, embed_initializer, maxlen, dtype)

DICT_CATEGORICAL = {
    'd0': [i for i in range(1, 11)],
    'd1': [i for i in range(1, 11)],
    'd2': [i for i in range(1, 11)],
    'd3': [i for i in range(1, 11)],
    'd4': [i for i in range(1, 11)],
    'd5': [i for i in range(1, 11)]
}

WDL_DEFAULT_VALUES = [
    [0],
    [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
    ['0'], ['0'], ['0'], ['0'], ['0'], ['0'],
    ['0:1.0;0:1.0;0:1.0'], ['0:1.0;0:1.0;0:1.0']
]

WDL_col_columns = [
    'label',
    's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11',
    'd0', 'd1', 'd2', 'd3', 'd4', 'd5',
    'm0', 'm1'
]

WDL_linear_feature_columns = [
    SparseFeat(name='d0', voc_size=12, use_hash=True, share_embed=None, embed_dim=4,
               embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),
    SparseFeat(name='d1', voc_size=12, use_hash=True, share_embed=None, embed_dim=4,
               embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),
    SparseFeat(name='d2', voc_size=12, use_hash=True, share_embed=None, embed_dim=4,
               embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),
    SparseFeat(name='d3', voc_size=12, use_hash=True, share_embed=None, embed_dim=4,
               embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),

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

WDL_nn_feature_columns = [
    SparseFeat(name='d4', voc_size=12, use_hash=True, share_embed=None, embed_dim=4,
               embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),
    SparseFeat(name='d5', voc_size=12, use_hash=True, share_embed=None, embed_dim=4,
               embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),

    VarLenSparseFeat(name='m0', voc_size=6, use_hash=True, weight_name='m0_weight', combiner='mean', embed_dim=4, maxlen=3,
                     embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string),
    VarLenSparseFeat(name='m1', voc_size=6, use_hash=True, weight_name='m1_weight', combiner='mean', embed_dim=4, maxlen=3,
                     embed_initializer=tf.keras.initializers.RandomUniform(minval=0.05, maxval=0.05, seed=0), dtype=tf.string)
]
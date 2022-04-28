import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

class EmbeddingLookup(Layer):
    def __init__(self, embedding, **kwargs):
        super(EmbeddingLookup, self).__init__(**kwargs)
        self.embedding = embedding

    def build(self, input_shape):
        super(EmbeddingLookup, self).build(input_shape)

    def call(self, inputs):
        idx = inputs
        embed = tf.nn.embedding_lookup(params=self.embedding, ids=idx)

        return embed


class EmbeddingLookupSparse(Layer):

    def __init__(self, embedding, has_weight=False, combiner='sum', **kwargs):
        super(EmbeddingLookupSparse, self).__init__(**kwargs)
        self.embedding  = embedding
        self.combiner   = combiner
        self.has_weight = has_weight

    def build(self, input_shape):
        super(EmbeddingLookupSparse, self).build(input_shape)

    def call(self, inputs):
        if self.has_weight:
            idx, val = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=val, combiner=self.combiner)
        else:
            idx = inputs
            combiner_embed = tf.nn.embedding_lookup_sparse(self.embedding, sp_ids=idx, sp_weights=None, combiner=self.combiner)

        return tf.expand_dims(combiner_embed, 1)


class VocabLayer(Layer):
    def __init__(self, keys, mask_value=None, **kwargs):
        super(VocabLayer, self).__init__(**kwargs)
        self.mask_value = mask_value
        vals = tf.range(2, len(keys) + 2)
        vals = tf.constant(vals, dtype=tf.int32)
        keys = tf.constant(keys)
        self.table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, vals), 1)

    def call(self, inputs):
        idx = self.table.lookup(inputs)
        if self.mask_value is not None:
            masks    = tf.not_equal(inputs, self.mask_value)
            paddings = tf.ones_like(idx) * (0)
            idx      = tf.where(masks, idx, paddings)

        return idx

class HashLayer(Layer):
    """
    hash the input to [0,num_buckets)
    if mask_zero = True,0 or 0.0 will be set to 0,other value will be set in range[1,num_buckets)
    """

    def __init__(self, num_buckets, mask_zero=False, **kwargs):
        self.num_buckets = num_buckets
        self.mask_zero = mask_zero
        super(HashLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(HashLayer, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        zero = tf.as_string(tf.zeros([1], dtype='int32'))
        num_buckets = self.num_buckets if not self.mask_zero else self.num_buckets - 1
        hash_x = tf.strings.to_hash_bucket_fast(x, num_buckets, name=None)
        if self.mask_zero:
            mask = tf.cast(tf.not_equal(x, zero), dtype='int64')
            hash_x = (hash_x + 1) * mask

        return hash_x

class Add(Layer):
    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Add, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            return inputs
        if len(inputs) == 1  :
            return inputs[0]
        if len(inputs) == 0:
            return tf.constant([[0.0]])
        return tf.keras.layers.add(inputs)


class DenseToSparseTensor(Layer):
    def __init__(self, mask_value=-1, **kwargs):
        super(DenseToSparseTensor, self).__init__()
        self.mask_value = mask_value

    def call(self, dense_tensor):
        idx = tf.where(tf.not_equal(dense_tensor, tf.constant(self.mask_value, dtype=dense_tensor.dtype)))
        sparse_tensor = tf.SparseTensor(idx, tf.gather_nd(dense_tensor, idx), tf.shape(dense_tensor, out_type=tf.int64))
        return sparse_tensor

    def get_config(self):
        config = super(DenseToSparseTensor, self).get_config()
        config.update({'mask_value': self.mask_value})
        return config

class FMLayer(Layer):

    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))

        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        concated_embeds_value = inputs
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term
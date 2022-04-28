import tensorflow as tf
from tensorflow.keras.layers import Layer

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

class WeightedSequenceLayer(Layer):

    def __init__(self, **kwargs):
        super(WeightedSequenceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(WeightedSequenceLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        key_input, value_input = inputs
        return tf.multiply(key_input, value_input)

class SequencePollingLayer(Layer):

    def __init__(self, mode='mean', **kwargs):
        self.mode = mode
        super(SequencePollingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SequencePollingLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.mode=='mean':
            return tf.keras.layers.GlobalAvgPool1D()(inputs)

        if self.mode=='max':
            return tf.keras.layers.GlobalMaxPool1D()(inputs)

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
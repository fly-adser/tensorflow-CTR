import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Zeros, glorot_normal, glorot_uniform
from tensorflow.keras.regularizers import l2
from src.model.tf_commonV2.feature_columns import reduce_sum
import tensorflow.keras.backend as K

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

class CrossNet(Layer):
    """The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.

      Input shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, units)``.

      Arguments
        - **layer_num**: Positive integer, the cross layer number

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix

        - **parameterization**: string, ``"vector"``  or ``"matrix"`` ,  way to parameterize the cross network.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, layer_num=2, parameterization='vector', l2_reg=0, seed=1024, **kwargs):
        self.layer_num = layer_num
        self.parameterization = parameterization
        self.l2_reg = l2_reg
        self.seed = seed
        print('CrossNet parameterization:', self.parameterization)
        super(CrossNet, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        dim = int(input_shape[-1])
        if self.parameterization == 'vector':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, 1),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        elif self.parameterization == 'matrix':
            self.kernels = [self.add_weight(name='kernel' + str(i),
                                            shape=(dim, dim),
                                            initializer=glorot_normal(
                                                seed=self.seed),
                                            regularizer=l2(self.l2_reg),
                                            trainable=True) for i in range(self.layer_num)]
        else:  # error
            raise ValueError("parameterization should be 'vector' or 'matrix'")
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(dim, 1),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(self.layer_num)]
        # Be sure to call this somewhere!
        super(CrossNet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        x_0 = tf.expand_dims(inputs, axis=2)
        x_l = x_0
        for i in range(self.layer_num):
            if self.parameterization == 'vector':
                xl_w = tf.tensordot(x_l, self.kernels[i], axes=(1, 0))
                dot_ = tf.matmul(x_0, xl_w)
                x_l = dot_ + self.bias[i] + x_l
            elif self.parameterization == 'matrix':
                xl_w = tf.einsum('ij,bjk->bik', self.kernels[i], x_l)  # W * xi  (bs, dim, 1)
                dot_ = xl_w + self.bias[i]  # W * xi + b
                x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
            else:  # error
                raise ValueError("parameterization should be 'vector' or 'matrix'")
        x_l = tf.squeeze(x_l, axis=2)
        return x_l

class CIN(Layer):
    """Compressed Interaction Network used in xDeepFM.This implemention is
    adapted from code that the author of the paper published on https://github.com/Leavingseason/xDeepFM.

      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)`` ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]`` if ``split_half=True``,else  ``sum(layer_size)`` .

      Arguments
        - **layer_size** : list of int.Feature maps in each layer.

        - **activation** : activation function used on feature maps.

        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.

        - **seed** : A Python integer to use as random seed.

      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024, **kwargs):
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")
        self.layer_size = layer_size
        self.split_half = split_half
        self.activation = activation
        self.l2_reg = l2_reg
        self.seed = seed
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [int(input_shape[1])]
        self.filters    = []
        self.bias       = []
        for i, size in enumerate(self.layer_size):

            self.filters.append(self.add_weight(name='filter' + str(i),
                                                shape=[1, self.field_nums[-1]
                                                       * self.field_nums[0], size],
                                                dtype=tf.float32, initializer=glorot_uniform(seed=self.seed + i),
                                                regularizer=l2(self.l2_reg)))

            self.bias.append(self.add_weight(name='bias' + str(i), shape=[size], dtype=tf.float32,
                                             initializer=tf.keras.initializers.Zeros()))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        self.activation_layers = [tf.keras.layers.Activation(self.activation) for _ in self.layer_size]

        super(CIN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):

        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))

        dim = int(inputs.get_shape()[-1])
        hidden_nn_layers = [inputs]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)

            dot_result_m = tf.matmul(
                split_tensor0, split_tensor, transpose_b=True)

            dot_result_o = tf.reshape(
                dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])

            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            curr_out = tf.nn.conv1d(
                dot_result, filters=self.filters[idx], stride=1, padding='VALID')

            curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.split_half:
                if idx != len(self.layer_size) - 1:
                    next_hidden, direct_connect = tf.split(
                        curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = reduce_sum(result, -1, keep_dims=False)

        return result
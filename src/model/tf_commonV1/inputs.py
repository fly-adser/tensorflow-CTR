from src.model.tf_commonV1.utils import *

class LoadData(object):

    def __init__(self, col_columns, feature_columns, DEFAULT_VALUES, batchSize=1024):
        self.col_columns     = col_columns
        self.feature_columns = feature_columns
        self.DEFAULT_VALUES  = DEFAULT_VALUES
        self.batchSize       = batchSize

    def _parse_function(self, example):
        item_feats = tf.io.decode_csv(example, record_defaults=self.DEFAULT_VALUES)
        parsed = dict(zip(self.col_columns, item_feats))
        feature_dict = {}
        for feat_col in self.feature_columns:
            if isinstance(feat_col, VarLenSparseFeat):
                if feat_col.weight_name is not None:
                    kvpairs = tf.strings.split([parsed[feat_col.name]], ';').values[:feat_col.maxlen]
                    kvpairs = tf.strings.split(kvpairs, ':')
                    kvpairs = kvpairs.to_tensor()
                    feat_ids, feat_vals = tf.split(kvpairs, num_or_size_splits=2, axis=1)
                    feat_ids = tf.reshape(feat_ids, shape=[-1])
                    feat_vals = tf.reshape(feat_vals, shape=[-1])
                    if feat_col.dtype != tf.string:
                        feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                    feat_vals = tf.strings.to_number(feat_vals, out_type=tf.float32)
                    feature_dict[feat_col.name] = feat_ids
                    feature_dict[feat_col.weight_name] = feat_vals
                else:
                    feat_ids = tf.strings.split([parsed[feat_col.name]], ',').values[:feat_col.maxlen]
                    feat_ids = tf.reshape(feat_ids, shape=[-1])
                    if feat_col.dtype != tf.string:
                        feat_ids = tf.strings.to_number(feat_ids, out_type=tf.int32)
                    feature_dict[feat_col.name] = feat_ids
            else:
                feature_dict[feat_col.name] = parsed[feat_col.name]

        label = parsed['label']

        return feature_dict, label

    def _paddingInfo(self):
        pad_shapes = {}
        pad_values = {}

        for feat_col in self.feature_columns:
            if isinstance(feat_col, VarLenSparseFeat):
                max_tokens = feat_col.maxlen
                pad_shapes[feat_col.name] = tf.TensorShape([max_tokens])
                pad_values[feat_col.name] = '0' if feat_col.dtype == 'string' else 0
                if feat_col.weight_name is not None:
                    pad_shapes[feat_col.weight_name] = tf.TensorShape([max_tokens])
                    pad_values[feat_col.weight_name] = tf.constant(1.0, dtype=tf.float32)

            elif isinstance(feat_col, SparseFeat):
                if feat_col.dtype == 'string':
                    pad_shapes[feat_col.name] = tf.TensorShape([])
                    pad_values[feat_col.name] = '0'
                else:
                    pad_shapes[feat_col.name] = tf.TensorShape([])
                    pad_values[feat_col.name] = 0
            else:
                if feat_col.dtype == 'string':
                    pad_shapes[feat_col.name] = tf.TensorShape([])
                    pad_values[feat_col.name] = '0.0'
                else:
                    pad_shapes[feat_col.name] = tf.TensorShape([])
                    pad_values[feat_col.name] = 0.0

        pad_shapes = (pad_shapes, (tf.TensorShape([])))
        pad_values = (pad_values, (tf.constant(0, dtype=tf.int32)))

        return pad_shapes, pad_values

    def load_data(self):
        pad_shapes, pad_values = self._paddingInfo()
        filenames = tf.data.Dataset.list_files(["../data/train.csv"])
        dataset_train = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
        dataset_train = dataset_train.map(self._parse_function, num_parallel_calls=60)
        dataset_train = dataset_train.padded_batch(batch_size=self.batchSize, padded_shapes=pad_shapes, padding_values=pad_values)

        filenames = tf.data.Dataset.list_files(["../data/valid.csv"])
        dataset_valid = filenames.flat_map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1))
        dataset_valid = dataset_valid.map(self._parse_function, num_parallel_calls=60)
        dataset_valid = dataset_valid.padded_batch(batch_size=self.batchSize, padded_shapes=pad_shapes, padding_values=pad_values)

        return dataset_train, dataset_valid
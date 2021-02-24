import numpy as np
import tensorflow as tf

from sparsemax import sparsemax

class TabNet(tf.keras.models.Model):
    """TabNet model class."""

    def __init__(self,
                 columns,
                 num_features,
                 feature_dim,
                 output_dim,
                 num_decision_steps,
                 relaxation_factor,
                 batch_momentum,
                 virtual_batch_size,
                 num_classes,
                 encoder_type='classification',
                 epsilon=0.00001):
        """Initializes a TabNet instance.

        Args:
          columns: The Tensorflow column names for the dataset.
          num_features: The number of input features (i.e the number of columns for
            tabular data assuming each feature is represented with 1 dimension).
          feature_dim: Dimensionality of the hidden representation in feature
            transformation block. Each layer first maps the representation to a
            2*feature_dim-dimensional output and half of it is used to determine the
            nonlinearity of the GLU activation where the other half is used as an
            input to GLU, and eventually feature_dim-dimensional output is
            transferred to the next layer.
          output_dim: Dimensionality of the outputs of each decision step, which is
            later mapped to the final classification or regression output.
          num_decision_steps: Number of sequential decision steps.
          relaxation_factor: Relaxation factor that promotes the reuse of each
            feature at different decision steps. When it is 1, a feature is enforced
            to be used only at one decision step and as it increases, more
            flexibility is provided to use a feature at multiple decision steps.
          batch_momentum: Momentum in ghost batch normalization.
          virtual_batch_size: Virtual batch size in ghost batch normalization. The
            overall batch size should be an integer multiple of virtual_batch_size.
          num_classes: Number of output classes.
          epsilon: A small number for numerical stability of the entropy calcations.

        Returns:
          A TabNet instance.
        """
        super(TabNet, self).__init__()

        self.columns = columns
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.batch_momentum = batch_momentum
        self.virtual_batch_size = virtual_batch_size
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.encoder_type = encoder_type

        self.feature_layer = tf.keras.layers.DenseFeatures(self.columns, name='feature_layer')
        self.feature_BN = tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)
        self.transform_f1_dense = tf.keras.layers.Dense(self.feature_dim * 2, name="Transform_f1", use_bias=False)
        self.transform_f2_dense = tf.keras.layers.Dense(self.feature_dim * 2, name="Transform_f2", use_bias=False)

        self.transform_f1_BN = []
        self.transform_f2_BN = []
        self.transform_f3_dense = []
        self.transform_f3_BN = []
        self.transform_f4_dense = []
        self.transform_f4_BN = []
        self.transform_coef = []
        self.transform_coef_BN = []
        for ni in range(self.num_decision_steps):
            self.transform_f1_BN.append(tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)) #,
                                                                           # virtual_batch_size=self.virtual_batch_size))
            self.transform_f2_BN.append(tf.keras.layers.BatchNormalization(momentum=self.batch_momentum))#,
                                                                           # virtual_batch_size=self.virtual_batch_size))
            self.transform_f3_dense.append(tf.keras.layers.Dense(self.feature_dim * 2,
                                                                 name="Transform_f3" + str(ni),
                                                                 use_bias=False))
            self.transform_f3_BN.append(tf.keras.layers.BatchNormalization(momentum=self.batch_momentum))#,
                                                                           # virtual_batch_size=self.virtual_batch_size))

            self.transform_f4_dense.append(tf.keras.layers.Dense(self.feature_dim * 2,
                                                                 name="Transform_f4" + str(ni),
                                                                 use_bias=False))
            self.transform_f4_BN.append(tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)) #,
                                                                           # virtual_batch_size=self.virtual_batch_size))
            self.transform_coef.append(tf.keras.layers.Dense(self.num_features,
                                                             name="Transform_coef" + str(ni),
                                                             use_bias=False))
            self.transform_coef_BN.append(tf.keras.layers.BatchNormalization(momentum=self.batch_momentum)) #,
                                                                             # virtual_batch_size=self.virtual_batch_size))
            if encoder_type == 'classification':
                self.encoder_output = tf.keras.layers.Dense(self.num_classes, use_bias=False)
            else:
                raise NotImplementedError

    def call(self, data, training=None):
        """TabNet encoder model."""

        # Reads and normalizes input features.
        features = self.feature_layer(data)
#         features = tf.clip_by_value(features, -1000, 1000)
        features = self.feature_BN(features, training=training)
        batch_size = tf.shape(features)[0]

        # Initializes decision-step dependent variables.
        output_aggregated = tf.zeros([batch_size, self.output_dim])
        masked_features = features
        mask_values = tf.zeros([batch_size, self.num_features])
        complemantary_aggregated_mask_values = tf.ones(
                [batch_size, self.num_features])
        aggregated_mask_values = tf.zeros([batch_size, self.num_features])
        total_entropy = 0
        aggregated_mask_values_all = []
        mask_values_all = []

        for ni in range(self.num_decision_steps):
            # TODO

        output = self.encoder_output(output_aggregated)
        if self.encoder_type:
            if self.num_classes > 1:
                output = tf.nn.softmax(output)
            else:
                output = tf.squeeze(tf.nn.sigmoid(output), axis=1)

        return output, output_aggregated, total_entropy, aggregated_mask_values_all, mask_values_all


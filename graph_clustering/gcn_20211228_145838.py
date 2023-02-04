from typing import Tuple
import tensorflow as tf


class GCN(tf.keras.layers.Layer):
  #Implementation of Graph Convolutional Network

  def __init__(self,
               n_channels,
               activation='selu',
               skip_connection = True):
    """Initializes the layer with specified parameters."""
    super(GCN, self).__init__()
    self.n_channels = n_channels
    self.skip_connection = skip_connection
    if isinstance(activation, str):
      self.activation = tf.keras.layers.Activation(activation)
    elif isinstance(tf.keras.layers.Activation):
      self.activation = activation
    elif activation is None:
      self.activation = tf.keras.layers.Lambda(lambda x: x)
    else:
      raise ValueError('GCN activation of unknown type')

  def build(self, input_shape):
    """Builds the Keras model according to the input shape."""
    self.n_features = input_shape[0][-1]
    self.kernel = self.add_variable(
        'kernel', shape=(self.n_features, self.n_channels))
    self.bias = self.add_variable('bias', shape=(self.n_channels,))
    if self.skip_connection:
      self.skip_weight = self.add_variable(
          'skip_weight', shape=(self.n_channels,))
    else:
      self.skip_weight = 0
    super().build(input_shape)

  def call(self, inputs):
    features, norm_adjacency = inputs
    print(norm_adjacency.dtype)

    assert isinstance(features, tf.Tensor)
    #assert isinstance(norm_adjacency, tf.SparseTensor)
    assert len(features.shape) == 2
    assert len(norm_adjacency.shape) == 2
    assert features.shape[0] == norm_adjacency.shape[0]

    output = tf.matmul(features, self.kernel)

    if self.skip_connection:
      output = output * self.skip_weight + tf.sparse.sparse_dense_matmul(
        norm_adjacency, output)
    else:
      output = tf.sparse.sparse_dense_matmul(norm_adjacency, output)
    output = output + self.bias
    return self.activation(output)
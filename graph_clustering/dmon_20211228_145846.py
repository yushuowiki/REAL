from typing import List
from typing import Tuple
import tensorflow as tf


class DMoN(tf.keras.layers.Layer):

  def __init__(self,
               n_clusters,
               collapse_regularization = 1,
               dropout_rate = 0.5,
               do_unpooling = True):
    """ layer initialization."""
    super(DMoN, self).__init__()
    self.n_clusters = n_clusters
    self.collapse_regularization = collapse_regularization
    self.dropout_rate = dropout_rate
    self.do_unpooling = do_unpooling

  def build(self, input_shape):
    self.transform = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            self.n_clusters,
            kernel_initializer='orthogonal',
            bias_initializer='zeros'),
        tf.keras.layers.Dropout(self.dropout_rate)
    ])
    super(DMoN, self).build(input_shape)

  def call(
      self, inputs):
    features, adjacency = inputs

    assert isinstance(features, tf.Tensor)
    assert isinstance(adjacency, tf.SparseTensor)
    assert len(features.shape) == 2
    assert len(adjacency.shape) == 2
    assert features.shape[0] == adjacency.shape[0]

    assignments = tf.nn.softmax(self.transform(features), axis=1)
    cluster_sizes = tf.math.reduce_sum(assignments, axis=0)  # Size [k].
    assignments_pooling = assignments / cluster_sizes  # Size [n, k].

    degrees = tf.sparse.reduce_sum(adjacency, axis=0)  # Size [n].
    degrees = tf.reshape(degrees, (-1, 1))

    number_of_nodes = adjacency.shape[1]
    number_of_edges = tf.math.reduce_sum(degrees)

    graph_pooled = tf.transpose(
        tf.sparse.sparse_dense_matmul(adjacency, assignments))
    graph_pooled = tf.matmul(graph_pooled, assignments)

    normalizer_left = tf.matmul(assignments, degrees, transpose_a=True)

    normalizer_right = tf.matmul(degrees, assignments, transpose_a=True)

    normalizer = tf.matmul(normalizer_left,
                           normalizer_right) / 2 / number_of_edges
    spectral_loss = -tf.linalg.trace(graph_pooled -
                                     normalizer) / 2 / number_of_edges
    self.add_loss(spectral_loss)

    collapse_loss = tf.norm(cluster_sizes) / number_of_nodes * tf.sqrt(
        float(self.n_clusters)) - 1
    self.add_loss(self.collapse_regularization * collapse_loss)

    features_pooled = tf.matmul(assignments_pooling, features, transpose_a=True)
    features_pooled = tf.nn.selu(features_pooled)
    if self.do_unpooling:
      features_pooled = tf.matmul(assignments_pooling, features_pooled)
    return features_pooled, assignments
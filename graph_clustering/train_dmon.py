from typing import Tuple
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from scipy import sparse
import dmon
import gcn
import utils
tf.compat.v1.enable_v2_behavior()
import graph_construct
from scipy.sparse import csr_matrix


def convert_scipy_sparse_to_sparse_tensor(
    matrix):
  #Converts a sparse matrix and converts it to Tensorflow SparseTensor.

  matrix = matrix.tocoo()
  return tf.sparse.SparseTensor(
      np.vstack([matrix.row, matrix.col]).T, matrix.data.astype(np.float32),
      matrix.shape)



def build_dmon(input_features,
               input_graph,
               input_adjacency):
  #Builds a Deep Modularity Network (DMoN) model from the Keras inputs.

  output = input_features
  for n_channels in [4096,1024,512]:   # FLAGS.architecture:
    output = gcn.GCN(n_channels)([output, input_graph])
  pool, pool_assignment = dmon.DMoN(
      n_clusters=1250,
      collapse_regularization=0.5,
      dropout_rate=0.5)([output, input_adjacency])
  return tf.keras.Model(
      inputs=[input_features, input_graph, input_adjacency],
      outputs=[pool, pool_assignment])


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  adjacency = sparse.load_npz('adjacency_nyc.npz')
  features = sparse.load_npz('features_nyc.npz')#原adj: sparse 现在: numpy
  features = features.todense()
  n_nodes = adjacency.shape[0]
  feature_size = features.shape[1]
  graph = convert_scipy_sparse_to_sparse_tensor(adjacency)
  graph_normalized = convert_scipy_sparse_to_sparse_tensor(utils.normalize_graph(adjacency.copy()))

  # Create model input placeholders of appropriate size
  input_features = tf.keras.layers.Input(shape=(feature_size,))
  input_graph = tf.keras.layers.Input((n_nodes,), sparse=True)
  input_adjacency = tf.keras.layers.Input((n_nodes,), sparse=True)

  model = build_dmon(input_features, input_graph, input_adjacency)

  # Computes the gradients wrt. the sum of losses, returns a list of them.
  def grad(model, inputs):
    with tf.GradientTape() as tape:
      _ = model(inputs, training=True)
      loss_value = sum(model.losses)
    return model.losses, tape.gradient(loss_value, model.trainable_variables)

  optimizer = tf.keras.optimizers.Adam(0.0001)#learning rate
  model.compile(optimizer, None)

  for epoch in range(25):#FLAGS.n_epochs
    loss_values, grads = grad(model, [features, graph_normalized, graph])
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f'epoch {epoch}, losses: ' +
          ' '.join([f'{loss_value.numpy():.6f}' for loss_value in loss_values]))

  # Obtain the cluster assignments.
  _, assignments = model([features, graph_normalized, graph], training=False)
  assignments = assignments.numpy()
  clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
  np.save('clusters_nyc_1250', clusters)


if __name__ == '__main__':
  app.run(main)
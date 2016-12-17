from mx_layers import *

class MLP:
  def __init__(self, shape, activation, X=None):
    self.activation  = activation
    self.shape       = shape
    self.X           = X

  def __call__(self, data_shape=None):
    inputs = variable('data') if self.X is None else self.X
    network = reduce(
      lambda symbol, d : activate(fully_connected(symbol, d), self.activation, data_shape),
      self.shape[:-1],
      inputs
    )
    network = fully_connected(network, self.shape[-1])
    network = softmax_loss(network)
    return network

import mxnet as mx
from MXLayers import *
from MXInitializer import DReLUInitializer
import numpy as np

class MLP:
  def __init__(self, shape, activation, initializer=DReLUInitializer()):
    self.activation  = activation
    self.shape       = shape
    self.initializer = initializer

  def __call__(self, data_shape):
    inputs = variable('data')
    network = reduce(
      lambda symbol, d : activate(fully_connected(symbol, d), self.activation, data_shape),
      self.shape[:-1],
      inputs
    )
    network = fully_connected(network, self.shape[-1])
    network = softmax(network)
    return network, self.initializer

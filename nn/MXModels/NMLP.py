from MXLayers import *

class CMLP:
  def __init__(self, shape, initializer):
    self.shape = shape
    self.initializer = initializer
  def __call__(self, data_shape):
    inputs = variable('data')
    outputs = fully_connected(inputs, self.shape[0])
    bound_shape = (1, self.shape[0])
    lower0 = variable('NDReLU_lower0', shape=bound_shape)
    upper0 = variable('NDReLU_upper0', shape=bound_shape)
    outputs = broadcast_maximum(lower0, outputs)
    outputs = broadcast_minimum(upper0, outputs)

    previous_lower = lower0
    previous_upper = upper0

    for i in range(len(self.shape) - 2):
      # fully-connected layer (including bias)
      weight_shape = (self.shape[i], self.shape[i + 1])
      weight = variable('NDReLU_weight%d' % (i + 1), shape=weight_shape)
      outputs = fully_connected(outputs, self.shape[i + 1], weight=weight)
      # calculate mean
      previous_mean = (previous_upper + previous_lower) / 2
      mean = fully_connected(previous_mean, self.shape[i + 1], weight=weight, no_bias=True)
      # center
      outputs = broadcast_minus(outputs, mean)
      # DReLU
      bound_shape = (1, self.shape[i + 1])
      lower = variable('NDReLU_lower%d' % (i + 1), shape=bound_shape)
      upper = variable('NDReLU_upper%d' % (i + 1), shape=bound_shape)
      outputs = broadcast_maximum(lower, outputs)
      outputs = broadcast_minimum(upper, outputs)
      # cache
      previous_lower = lower
      previous_upper = upper

    outputs = fully_connected(outputs, self.shape[-1])
    outputs = softmax(outputs)
    return outputs, self.initializer

class NMLP:
  def __init__(self, shape, initializer):
    self.shape = shape
    self.initializer = initializer
  def __call__(self, data_shape):
    inputs = variable('data')
    outputs = fully_connected(inputs, self.shape[0])
    bound_shape = (1, self.shape[0])
    lower0 = variable('NDReLU_lower0', shape=bound_shape)
    upper0 = variable('NDReLU_upper0', shape=bound_shape)
    outputs = broadcast_maximum(lower0, outputs)
    outputs = broadcast_minimum(upper0, outputs)

    previous_lower = lower0
    previous_upper = upper0

    for i in range(len(self.shape) - 2):
      # fully-connected layer (including bias)
      weight_shape = (self.shape[i], self.shape[i + 1])
      weight = variable('NDReLU_weight%d' % (i + 1), shape=weight_shape)
      outputs = fully_connected(outputs, self.shape[i + 1], weight=weight, no_bias=True)
      # calculate mean
      previous_mean = (previous_upper + previous_lower) / 2
      mean = fully_connected(previous_mean, self.shape[i + 1], weight=weight, no_bias=True)
      # center
      outputs = broadcast_minus(outputs, mean)
      # calculate deviation
      weight_sign = mx.symbol.sign(weight)
      # rescale
      # DReLU
      bound_shape = (1, self.shape[i + 1])
      lower = variable('NDReLU_lower%d' % (i + 1), shape=bound_shape)
      upper = variable('NDReLU_upper%d' % (i + 1), shape=bound_shape)
      outputs = broadcast_maximum(lower, outputs)
      outputs = broadcast_minimum(upper, outputs)
      # cache
      previous_lower = lower
      previous_upper = upper

    outputs = fully_connected(outputs, self.shape[-1])
    outputs = softmax(outputs)
    return outputs, self.initializer

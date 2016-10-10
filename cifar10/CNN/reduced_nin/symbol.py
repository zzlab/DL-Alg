import sys
sys.path.append('../../../nn')

class ReducedNIN:
  def __init__(self, activation, filter_count, initializer):
    self.activation   = activation
    self.filter_count = filter_count
    self.initializer  = initializer
  def __call__(self, data_shape, mediate=False):
    import MXLayers as layers
    def __convolution(*args, **kwargs):
      return layers.convolution(*args, activation=self.activation, data_shape=data_shape, **kwargs)
    inputs = layers.variable('data')
    if mediate:
      activations = []
    outputs = __convolution(inputs, (3, 3), self.filter_count, pad=(1, 1))
    if mediate:
      activations.append(outputs)
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    if mediate:
      activations.append(outputs)
    outputs = layers.pooling(outputs, 'max', (2, 2), (2, 2))
    outputs = __convolution(outputs, (1, 1), self.filter_count, pad=(1, 1))
    if mediate:
      activations.append(outputs)
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    if mediate:
      activations.append(outputs)
    outputs = layers.pooling(outputs, 'max', (2, 2), (2, 2))
    outputs = __convolution(outputs, (1, 1), self.filter_count, pad=(1, 1))
    if mediate:
      activations.append(outputs)
    outputs = __convolution(outputs, (1, 1), self.filter_count)
    if mediate:
      activations.append(outputs)
    outputs = layers.pooling(outputs, 'avg', (8, 8), (8, 8))
    outputs = layers.flatten(outputs)
    outputs = layers.softmax(outputs)
    if mediate:
      return mx.Group(outputs)
    return outputs, self.initializer 

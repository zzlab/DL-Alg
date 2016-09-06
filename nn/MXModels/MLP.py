from MXLayers import *

class MLP:
  def __init__(self, shape, activation, initializer):
    self.activation  = activation
    self.initializer = initializer
    self.shape       = shape

  def __call__(self, data_shape, intermediate_result=False):
    inputs = variable('data')
    if not intermediate_result:
      network = reduce(
        lambda symbol, dimension : activate(fully_connected(symbol, dimension), self.activation, data_shape),
        self.shape[:-1],
        inputs
      )
      network = fully_connected(network, self.shape[-1])
      return softmax(network), self.initializer
    else:
      output_group = []
      def layer(symbol, dimension):
        outputs = activate(fully_connected(symbol, dimension), self.activation, data_shape)
        output_group.append(outputs)
        return outputs
      network = reduce(
        layer,
        self.shape[:-1],
        inputs
      )
      network = fully_connected(network, self.shape[-1])
      output_group.append(network)
      return mx.symbol.Group(output_group), self.initializer

from MXLayers import *

class CNN:
  def __init__(self, settings, activation, initializer, output_shape=10):
    self.activation   = activation
    self.initializer  = initializer
    self.settings     = settings
    self.output_shape = output_shape

  def __call__(self, data_shape, intermediate_result=False):
    inputs = variable('data')
    if not intermediate_result:
      network = reduce(
        lambda symbol, settings : activate(convolution(symbol, *settings), self.activation, data_shape),
        self.settings,
        inputs
      )
      network = fully_connected(network, self.output_shape)
      return softmax(network), self.initializer
    else:
      output_group = []
      def layer(symbol, settings):
        outputs = activate(convolution(symbol, *settings), self.activation, data_shape)
        output_group.append(outputs)
        return outputs
      network = reduce(
        layer,
        self.settings,
        inputs
      )
      network = flatten(network)
      network = fully_connected(network, self.output_shape)
      output_group.append(network)
      return mx.symbol.Group(output_group), self.initializer

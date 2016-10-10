import mxnet as mx
from MXLayers import *
from MXInitializer import DReLUInitializer
import numpy as np

def mean(array):
  return array.asnumpy().mean()

def deviation(array):
  return array.asnumpy().std()

class MLP:
  def __init__(self, shape, activation, input_data=None):
    self.activation  = activation
    self.shape       = shape
    self.input_data  = input_data

  def sequentially_initialize(self, output_group, initializer):
    device = mx.cpu()
      
    network = output_group[-1]
    arguments = network.list_arguments()
    states = network.list_auxiliary_states()

    argument_shapes, _, state_shapes = network.infer_shape(data=self.input_data.shape)

    argument_shapes = dict(zip(arguments, argument_shapes))
    state_shapes = dict(zip(states, state_shapes))

    args = {}
    for argument, shape in argument_shapes.items():
      if 'weight' not in argument:
        args[argument] = mx.nd.ones(shape)
        initializer(argument, args[argument])
    args['data'] = mx.nd.array(self.input_data, device)

    auxes = {state : mx.nd.zeros(shape) for state, shape in state_shapes}
    for state, array in auxes.items():
      initializer(state, array)

    E_X_2 = mean(args['data'] ** 2)

    print 'data', deviation(args['data'])
    for index, symbol in enumerate(output_group):
      weight = 'fullyconnected%d_weight' % index
      weight_shape = argument_shapes[weight]
      std = 1 / (weight_shape[0] * E_X_2) ** 0.5
      args[weight] = mx.nd.array(np.random.normal(0, std, weight_shape), device)
      executor = symbol.bind(device, args, aux_states=auxes)
      executor.forward()
      print 'std', deviation(executor.outputs[0])
      E_X_2 = mean(executor.outputs[0] ** 2)
  
    del args['data']
    return DReLUInitializer(dictionary=args)

  def __call__(self, data_shape, intermediate_result=False):
    output_group = []
    def layer_forward(symbol, dimension):
      outputs = activate(fully_connected(symbol, dimension), self.activation, data_shape)
      output_group.append(outputs)
      return outputs

    # compose symbol
    inputs = variable('data')
    network = reduce(
      layer_forward,
      self.shape[:-1],
      inputs
    )
    network = fully_connected(network, self.shape[-1])

    initializer = DReLUInitializer()

    # sequential initialization
    if self.input_data is not None:
      initializer = self.sequentially_initialize(output_group, initializer)

    if not intermediate_result:
      return softmax(network), initializer
    else:
      output_group.append(network)
      return mx.symbol.Group(output_group), initializer

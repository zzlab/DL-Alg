from minpy.nn.model_builder import *
from minpy.array import Array
import minpy.numpy as np
from minpy.numpy import prod as product

def to_float(array):
  if array.shape == (1,):
    return array.asnumpy()[0]

def to_np(array):
  return array.asnumpy()

def array_mean(array, axis=None):
  import numpy
  if isinstance(array, Array):
    array = array.asnumpy()
  return numpy.mean(array, axis)

def array_std(array, axis=None):
  import numpy
  if isinstance(array, Array):
    array = array.asnumpy()
  return numpy.std(array, axis)

def np_max(array, axis=None):
  import numpy
  if isinstance(array, Array):
    array = array.asnumpy()
  return numpy.max(array, axis)

def np_min(array, axis=None):
  import numpy
  if isinstance(array, Array):
    array = array.asnumpy()
  return numpy.min(array, axis)

def np_abs(array):
  import numpy
  if isinstance(array, Array):
    array = array.asnumpy()
  return numpy.abs(array)

def rescale(container, inputs, parameters):
  """ recover original distribution at the final layer of every container. """
  # returns outputs, factor list

  factors = []
  all_factors = []
  input_shape = inputs.shape[1:]

  # find final affine layer
  ending = None
  for index in range(len(container._modules) - 1, -1, -1):
    value = container._modules[index]
    if isinstance(value, Affine) or isinstance(value, Convolution):
      ending = index
      break

  # iterate through module
  for module_index, module in enumerate(container._modules):
    shapes = module.parameter_shape(input_shape)
    input_shape = module.output_shape(input_shape)
    if isinstance(module, Affine) or isinstance(module, Convolution):
      for key, value in shapes.items():
        if 'weight' in key:
          E_X_2 = np.mean(inputs ** 2)
          if isinstance(module, Affine):
            n = value[0]
          else:
            C, W, H = value[1:]
            n = C * W * H
          std_from = np.std(parameters[key])
          std_to = 1 / (E_X_2 * n) ** 0.5
          rescaling_factor = std_to / std_from
          if module_index == ending:
            parameters[key] /= np.prod(np.array(factors))
          else:
            factors.append(rescaling_factor)
            parameters[key] *= rescaling_factor
          '''
          factors.append(rescaling_factor)
          parameters[key] *= rescaling_factor
          '''

    inputs = module.forward(inputs, parameters)

  return inputs, factors

def affine_rescale(container, inputs, parameters, epsilon=1E-3):
  input_shape = inputs.shape[1:]

  # iterate through module
  for module_index, module in enumerate(container._modules):
    shapes = module.parameter_shape(input_shape)
    input_shape = module.output_shape(input_shape)
    if isinstance(module, Affine):
      weight = module._weight
      bias = module._bias

#     print 'pre', array_std(parameters[weight])
      outputs = module.forward(inputs, parameters)
      std = array_std(outputs, axis=0)
      while epsilon < np_max(np_abs(std - 1)):
        parameters[weight] /= std
        outputs = module.forward(inputs, parameters)
        std = array_std(outputs, axis=0)
#     print 'post', array_std(parameters[weight])
      parameters[weight] *= 1.6

      inputs = outputs
#     inputs = module.forward(inputs, parameters)

  return inputs

from minpy.nn.model_builder import *
import minpy.numpy as np

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

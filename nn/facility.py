import numpy as np0
import mxnet.symbol as symbol
from minpy.core import Function
from minpy.nn.model_builder import *
from minpy.array import Array
import minpy.numpy as np
from minpy.numpy import prod as product

def load_mnist(path='/home/alex/experiment/mnist/utilities', shape=None):
  import cPickle,gzip
  import minpy.numpy as np
  with gzip.open(path+'/mnist.gz', 'rb') as data:
    package = cPickle.load(data)
  if shape is not None:
    package = list(package) 
    for index, data in enumerate(package):
      X, Y = data
      N, D = X.shape
      package[index] = (X.reshape((N,) + shape), Y)
    package = tuple(package)
  unpacked = []
  for data in package:
    unpacked.extend(data)
  unpacked = tuple(unpacked)
  return unpacked

def batch_dot(left, right):
  # assert left.shape[0] == right.shape[0] and left.shape[2] == right.shape[1]
  left_symbol = symbol.Variable('left')
  right_symbol = symbol.Variable('right')
  result_symbol = symbol.batch_dot(left_symbol, right_symbol)
  shapes = {'left' : left.shape, 'right' : right.shape}
  kwargs = {'left' : left, 'right' : right}
  return Function(result_symbol, shapes)(**kwargs)

def batch_scalar_product(left, right):
  left_shape, right_shape = map(int, left.shape), map(int, right.shape)
  # assert left_shape == right_shape
  N, D = left_shape
  left = left.reshape((N, 1, D))
  right = right.reshape((N, D ,1))
  result = batch_dot(left, right)
  result = result.reshape((N, 1))
  return result

def mark():
  from inspect import currentframe, getframeinfo
  frame = currentframe().f_back
  string = 'at %s %s' % (getframeinfo(frame).filename, frame.f_lineno)
# print '*' * len(string)
  print string
# print '*' * len(string)

def identity(n):
  array = np.zeros((n, n))
  array[np.arange(n), np.arange(n)] = 1
  return array

def diagonal(array):
  rows, columns = array.shape
  assert rows == columns
  mask = identity(rows)
  result = array * mask
  return result

def outer(left, right):
  # left and right must be vectors
  left = left.reshape((1, max(left.shape)))
  right = right.reshape((max(right.shape), 1))
  product = left * right
  return product

def np_wrapper(f):
  def wrapped(*args, **kwargs):
    args = [arg.asnumpy() if isinstance(arg, Array) else arg for arg in args]
    kwargs = { key : value.asnumpy() if isinstance(value, Array) else value for key, value in kwargs.items() }
    return f(*args, **kwargs)
  return wrapped

def average_top_k(table, k):
  return sum(sorted(table)[len(table) - k:]) / float(k)

@np_wrapper
def accuracy(p, labels):
  N, D = p.shape
  return np0.sum(np0.argmax(p, axis=1) == labels) / float(N)

def softmax_probability(p, channel):
  N, C = p.shape
  p -= np.max(p, axis=1).reshape((N, 1))
  code = np.zeros((N, C))
  np.onehot_encode(channel, code)
  p = np.exp(p)
  selected_p = p * code
  total_p = np.sum(p, axis=1).reshape((N, 1))
  return np.sum(selected_p / total_p, axis=1)

def KL(P, Q):
  return np.sum(P * np.log(P / Q), axis=1)

def flatten(array):
  return array.asnumpy().flatten()

def to_float(array):
  if array.shape == (1,):
    return array.asnumpy()[0]

def clip(X, lower, upper):
  return np.minimum(upper, np.maximum(lower, X))

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

if __name__ == '__main__':
  left = np0.random.normal(0, 1, (100, 1, 10))
  right = np0.random.normal(0, 1, (100, 10, 10))
  print batch_dot(left, right).shape

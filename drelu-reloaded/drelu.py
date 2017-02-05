from mx_initializer import PReLUInitializer
from mx_layers import flatten, variable
from mxnet.symbol import broadcast_minimum, broadcast_maximum

_n_drelus = 0
def drelu(X, shape):
  _, input_shape, _ = X.infer_shape(**shape)
  input_shape = input_shape[0]
  if len(input_shape) is 2: bound_shape = input_shape[1:]
  elif len(input_shape) is 4: bound_shape = (1, input_shape[1], 1, 1)
  global _n_drelus
  lower = variable('drelu%d_lower_bound' % _n_drelus, shape=bound_shape)
  upper = variable('drelu%d_upper_bound' % _n_drelus, shape=bound_shape)
  _n_drelus += 1
  return broadcast_minimum(upper, broadcast_maximum(lower, X))

class DReLUInitializer(PReLUInitializer):
  def __init__(self, lower, upper):
    super(DReLUInitializer, self).__init__()
    self._lower, self._upper = lower, upper
  def __call__(self, identifier, array):
    if 'lower' in identifier: array[:] = self._lower
    elif 'upper' in identifier: array[:] = self._upper
    else: super(DReLUInitializer, self).__call__(identifier, array)

if __name__ is '__main__':
# X_SHAPE = (10000, 3072)
  X_SHAPE = (10000, 3, 32, 32)
  X = variable('data')
  network = drelu(X, {'data' : X_SHAPE})
  args = network.list_arguments()
  arg_shapes, output_shapes, _ = network.infer_shape(data=X_SHAPE)
  print dict(zip(args, arg_shapes))
  print output_shapes

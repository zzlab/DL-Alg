import mxnet as mx

def _map_args(args, mapping):
  return {mapping.get(key, key) : value for key, value in args.items()}

def ReLU(X, **kwargs):
  return mx.sym.Activation(data=X, act_type='relu', **kwargs)

def activate(*args, **kwargs):
  return __activate()(*args, **kwargs)

class __activate(object):
  count = {}
  def __call__(self, inputs, mode, data_shape=None):
    try:
      self.__class__.count[mode] += 1
    except:
      self.__class__.count.update({mode : 0})

    if mode == 'ReLU':
      return mx.symbol.Activation(data=inputs, act_type='relu')

    if mode == 'ReLUBN':
      return mx.symbol.BatchNorm(
        mx.symbol.Activation(data=inputs, act_type='relu'),
        fix_gamma=False
      )

    if mode == 'BNReLU':
      return mx.symbol.Activation(
        data     = mx.symbol.BatchNorm(inputs, fix_gamma=False),
        act_type = 'relu'
      )

    if mode == 'DReLU':
      label = 'DReLU%d' % self.__class__.count[mode]
      _, shape, _ = inputs.infer_shape(data=data_shape)
      return DReLU(inputs, shape[0], label)

    if mode == 'CDReLU':
      label = 'CDReLU%d' % self.__class__.count[mode]
      _, shape, _ = inputs.infer_shape(data=data_shape)
      return CDReLU(inputs, shape[0], label)

    if mode == 'NDReLU':
      label = 'NDReLU%d' % self.__class__.count[mode]
      _, shape, _ = inputs.infer_shape(data=data_shape)
      return NDReLU(inputs, shape[0], label)

    if mode == 'BNDReLU':
      label = 'BNDReLU%d' % self.__class__.count[mode]
      _, shape, _ = inputs.infer_shape(data=data_shape)
      outputs = mx.symbol.BatchNorm(inputs, fix_gamma=False)
      outputs = DReLU(outputs, shape[0], label)
      return outputs

    if mode == 'DReLUBN':
      label = 'DReLUBN%d' % self.__class__.count[mode]
      _, shape, _ = inputs.infer_shape(data=data_shape)
      outputs = DReLU(inputs, shape[0], label)
      return mx.symbol.BatchNorm(outputs, fix_gamma=False)

    if mode == 'BGNDReLU':
      label = 'BGNDReLU%d' % self.__class__.count[mode]
      _, shape, _ = inputs.infer_shape(data=data_shape)
      outputs = BGNDReLU(inputs, shape[0], label)
      return outputs

    raise Exception()

def DReLU(inputs, input_shape, label):
  if len(input_shape) == 2:
    print 'DReLU'
    bound_shape = (1, input_shape[1])
  elif len(input_shape) == 4:
    print 'spatial DReLU'
    bound_shape = (1, input_shape[1], 1, 1)
  else:
    raise Exception()

  lower = variable('%s_lower' % label, shape=bound_shape)
  upper = variable('%s_upper' % label, shape=bound_shape)
  outputs = broadcast_minimum(upper, broadcast_maximum(lower, inputs))
  return outputs

def CDReLU(inputs, input_shape, label):
  if len(input_shape) == 2:
    print 'CDReLU'
    bound_shape = (1, input_shape[1])
  elif len(input_shape) == 4:
    print 'spatial CReLU'
    bound_shape = (1, input_shape[1], 1, 1)
  else:
    raise Exception()

  lower = variable('%s_lower' % label, shape=bound_shape)
  upper = variable('%s_upper' % label, shape=bound_shape)
  outputs = broadcast_minimum(upper, broadcast_maximum(lower, inputs))
  outputs = broadcast_minus(outputs, (lower + upper) / 2)
  return outputs

def NDReLU(inputs, input_shape, label, epsilon=0.001):
  if len(input_shape) == 2:
    print 'NDReLU'
    bound_shape = (1, input_shape[1])
    parameter_shape = (1, input_shape[1])
  elif len(input_shape) == 4:
    print 'spatial NDReLU'
    bound_shape = (1, input_shape[1], 1, 1)
    parameter_shape = (1, input_shape[1], 1, 1)
  else:
    raise Exception()

  gamma = variable('%s_gamma' % label, shape=parameter_shape)
  beta = variable('%s_beta' % label, shape=parameter_shape)

  lower = variable('%s_lower' % label, shape=bound_shape)
  upper = variable('%s_upper' % label, shape=bound_shape)

  outputs = broadcast_minimum(upper, broadcast_maximum(lower, inputs))

  outputs = broadcast_minus(outputs, (lower + upper) / 2)
  outputs = broadcast_divide(outputs, (upper - lower) + epsilon)

  outputs = broadcast_multiply(outputs, gamma)
  outputs = broadcast_plus(outputs, beta)

  return outputs

def BGNDReLU(inputs, input_shape, label, magnitude=1.6, epsilon=0.001):
  if len(input_shape) == 2:
    bound_shape = (1, input_shape[1])
    parameter_shape = (1, input_shape[1])
  elif len(input_shape) == 4:
    bound_shape = (1, input_shape[1], 1, 1)
    parameter_shape = (1, input_shape[1], 1, 1)
  else:
    raise Exception()

  lower = variable('%s_lower' % label, shape=bound_shape)
  upper = variable('%s_upper' % label, shape=bound_shape)

  outputs = broadcast_minimum(upper, broadcast_maximum(lower, inputs))

  mean = block_gradient((lower + upper) / 2)
  deviation = block_gradient((upper - lower) / magnitude)
  outputs = broadcast_minus(outputs, mean)
  outputs = broadcast_divide(outputs, deviation + epsilon)

  return outputs

def NDReLUFC(*args, **kwargs):
  return __NDReLUFC()(*args, **kwargs)

class __NDReLUFC(object):
  count = 0

  def __call__(self, inputs, n):
    outputs = fully_connected(inputs, no_bias=True)

    _, shape, _ = outputs.infer_shape(data=data_shape)
    shape = shape[0]
    parameter_shape = (1, shape[1])
    label = 'NDReLUFC%d' % self.__class__.count
    gamma = variable('%s_gamma' % label, shape=parameter_shape)
    beta  = variable('%s_beta' % label, shape=parameter_shape)
    lower = variable('%s_lower' % label, shape=parameter_shape)
    upper = variable('%s_upper' % label, shape=parameter_shape)

    outputs = broadcast_minimum(upper, broadcast_maximum(lower, outputs))
    outputs = broadcast_divide(outputs, upper - lower)
    outputs = broadcast_multiply(outputs, gamma)
    outputs = broadcast_plus(outputs, beta)

    self.__class__.count += 1

    return outputs
 
def NDReLUConvolution(*args, **kwargs):
  return __NDReLUConvolution()(*args, **kwargs)

class __NDReLUConvolution(object):
  count = 0

  def __call__(self, inputs, kernel, filters, data_shape, stride=(1, 1), pad=(0, 0)):
    outputs = convolution(inputs, kernel, filters, stride, pad, no_bias=True)

    _, shape, _ = outputs.infer_shape(data=data_shape)
    shape = shape[0]
    parameter_shape = (1, shape[1], 1, 1)
    label = 'NDReLUConvolution%d' % self.__class__.count
    gamma = variable('%s_gamma' % label, shape=parameter_shape)
    beta  = variable('%s_beta' % label, shape=parameter_shape)
    lower = variable('%s_lower' % label, shape=parameter_shape)
    upper = variable('%s_upper' % label, shape=parameter_shape)

    outputs = broadcast_minimum(upper, broadcast_maximum(lower, outputs))
    outputs = broadcast_divide(outputs, upper - lower)
    outputs = broadcast_multiply(outputs, gamma)
    outputs = broadcast_plus(outputs, beta)

    self.__class__.count += 1

    return outputs
  
def terminate_gradient(X):
  return mx.symbol.BlockGrad(data=X)

def batch_normalization(X, **kwargs):
  mapping = {
    'id'           : 'name',
  }
  return mx.symbol.BatchNorm(data=X, **_map_args(kwargs, mapping))

def broadcast(X, shape):
  return mx.symbol.broadcast_to(X, shape=shape)

def broadcast_axis(inputs, axis, size):
  return mx.sym.broadcast_axis(inputs, axis=axis, size=size)

def broadcast_maximum(lower, inputs):
  minus = mx.sym.broadcast_minus(inputs, lower)
  sign  = mx.sym.sign(minus)
  return inputs * mx.sym.maximum(0, sign) - mx.sym.broadcast_mul(lower, mx.sym.minimum(0, sign))

def broadcast_minimum(upper, inputs):
  minus = mx.sym.broadcast_minus(inputs, upper)
  sign  = mx.sym.sign(minus)
  return mx.sym.broadcast_mul(upper, mx.sym.maximum(0, sign)) - inputs * mx.sym.minimum(0, sign)

def broadcast_plus(left, right):
  return mx.sym.broadcast_plus(left, right)

def broadcast_minus(left, right):
  return mx.sym.broadcast_minus(left, right)

def broadcast_multiply(left, right):
  return mx.sym.broadcast_mul(left, right)

def broadcast_divide(left, right):
  return mx.sym.broadcast_div(left, right)

def concatenate(**kwargs):
  mapping = {'axis' : 'dim', 'n_inputs' : 'num_args'}
  return mx.symbol.Concat(*kwargs.pop('X'), **_map_args(kwargs, mapping))

def convolution(**kwargs):
  mapping = {
    'attribute'    : 'attr',
    'cudnn_mode'   : 'cudnn_tune',
    'id'           : 'name',
    'kernel_shape' : 'kernel',
    'n_filters'    : 'num_filter',
    'n_groups'     : 'num_group',
    'X'            : 'data'
  }
  return mx.symbol.Convolution(**_map_args(kwargs, mapping))

def dot(left, right):
  return mx.symbol.dot(left, right)

def dropout(X, p):
  return mx.symbol.Dropout(X, p=p)

def flatten(inputs, **kwargs):
  return mx.symbol.Flatten(inputs, **kwargs)

def fully_connected(**kwargs):
  mapping = {'attribute' : 'attr', 'id' : 'name', 'n_hidden_units' : 'num_hidden', 'X' : 'data'}
  return mx.symbol.FullyConnected(**_map_args(kwargs, mapping))

def group(symbols):
  return mx.symbol.Group(symbols)

def linear_regression_loss(*args, **kwargs):
  mapping = {'id' : 'name'}
  return mx.symbol.LinearRegressionOutput(*args, **_map_args(kwargs, mapping))

def maximum(left, right):
  return mx.sym.maximum(left, right) 

def mean(*args, **kwargs):
  return mx.symbol.mean(*args, **kwargs)

def minimum(left, right):
  return mx.sym.minimum(left, right)
 
def norm(array):
  return mx.symbol.norm(array)

def pad(X, width, mode, value=0):
  # temporary solution
  D = len(width) / 2
  for i in range(D):
    if width[i * 2] + width[i * 2 + 1] > 0:
      X = swap_axes(X, i, D - 1)
      X = mx.symbol.Pad(data=X, pad_width=(0,) * 6 + width[i * 2 : i * 2 + 2], mode=mode, constant_value=value)
      X = swap_axes(X, i, D - 1)
  return X
  
def pooling(**kwargs):
  mode_mapping = {'average' : 'avg', 'maximum' : 'max'}
  kwargs['mode'] = mode_mapping[kwargs['mode']]
  mapping = {'kernel_shape' : 'kernel', 'mode' : 'pool_type', 'X' : 'data'}
  return mx.symbol.Pooling(**_map_args(kwargs, mapping))

def reshape(X, shape, **kwargs):
  return mx.symbol.Reshape(data=X, shape=shape, **kwargs)

def sigmoid(X):
  return mx.sym.Activation(data=X, act_type='sigmoid')

def sign(X):
  # -1 0 1
  return mx.symbol.sign(X)

def slice(**kwargs):
  mapping = {'X' : 'data', 'n_outputs' : 'num_outputs'}
  return mx.symbol.SliceChannel(**_map_args(kwargs, mapping))

def softmax(X):
  return mx.symbol.SoftmaxActivation(data=X, mode='instance')

def softmax_loss(*args, **kwargs):
  mapping = {'prediction' : 'data', 'id' : 'name'}
  return mx.symbol.SoftmaxOutput(**_map_args(kwargs, mapping))

def sum(*args, **kwargs):
  return mx.symbol.sum(*args, **kwargs)

def swap_axes(inputs, left, right):
  return mx.sym.SwapAxis(data=inputs, dim1=left, dim2=right)

def tanh(X):
  return mx.sym.Activation(data=X, act_type='tanh')

def uniform(*args, **kwargs):
  return mx.symbol.uniform(*args, **kwargs)

def variable(label, **kwargs):
  return mx.symbol.Variable(label, **kwargs)

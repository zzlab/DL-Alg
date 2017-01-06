import mxnet as mx

def ReLU(X):
  return mx.sym.Activation(data=X, act_type='relu')

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
  
def block_gradient(inputs):
  return mx.symbol.BlockGrad(data=inputs)

def batch_normalization(inputs, **kwargs):
  return mx.symbol.BatchNorm(data=inputs, **kwargs)

def broadcast(inputs, shape):
  return mx.sym.broadcast_to(inputs, shape)

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

def convolution(inputs, kernel, filters, stride=(1, 1), pad=(0, 0), activation=None, **kwargs):
  if activation:
    return activate(
      mx.symbol.Convolution(data=inputs, kernel=kernel, num_filter=filters, stride=stride, pad=pad),
      activation,
      **kwargs
    )
  else:
    return mx.symbol.Convolution(data=inputs, kernel=kernel, num_filter=filters, stride=stride, pad=pad)

def dropout(inputs, ratio):
  return mx.symbol.Dropout(inputs, p=ratio)

def flatten(inputs):
  return mx.symbol.Flatten(inputs)

def fully_connected(inputs, n, **kwargs):
  return mx.symbol.FullyConnected(data=inputs, num_hidden=n, **kwargs)

def maximum(left, right):
  return mx.sym.maximum(left, right) 

def minimum(left, right):
  return mx.sym.minimum(left, right)
 
def pooling(inputs, mode, kernel, stride=(1, 1), pad=(0, 0)):
  mapping = {'average' : 'avg', 'maximum' : 'max'}
  return mx.symbol.Pooling(inputs, pool_type=mapping[mode], kernel=kernel, stride=stride, pad=pad)

def reshape(inputs, shape, **kwargs):
  return mx.symbol.Reshape(data=inputs, target_shape=shape, **kwargs)

def softmax_activation(X):
  return mx.symbol.SoftmaxActivation(data=X, mode='instance')

def softmax_loss(inputs, labels=None):
  return mx.symbol.SoftmaxOutput(data=inputs, label=labels) if labels \
    else mx.symbol.SoftmaxOutput(data=inputs, name='softmax')

def swap_axes(inputs, left, right):
  return mx.sym.SwapAxis(data=inputs, dim1=left, dim2=right)

def variable(label, **kwargs):
  return mx.symbol.Variable(label, **kwargs)

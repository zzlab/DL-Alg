import minpy.numpy as np
import minpy.nn.model_builder as builder
import numpy as np0

from facility import *

class DReLU(builder.Module):
  _count = 0
  def __init__(self, lower=0, upper=1):
    super(DReLU, self).__init__()
    self._lower = 'DReLU%d_lower' % DReLU._count
    self._upper = 'DReLU%d_upper' % DReLU._count
    self._initial_lower = lower
    self._initial_upper = upper
    DReLU._count += 1

  def forward(self, inputs, parameters):
    lower = parameters[self._lower]
    upper = parameters[self._upper]
    return np.minimum(upper, np.maximum(lower, inputs))

  def output_shape(self, input_shape):
    return input_shape

  def parameter_settings(self):
    return {
      self._lower : { 'init_rule' : 'constant', 'init_config' : { 'value' : self._initial_lower} },
      self._upper : { 'init_rule' : 'constant', 'init_config' : { 'value' : self._initial_upper} },
    }
  
  def parameter_shape(self, input_shape):
    if len(input_shape) == 1:
      return {
        self._lower : input_shape,
        self._upper : input_shape,
      }
    if len(input_shape) == 3:
      return {
        self._lower : (1,) + input_shape[:1] + (1, 1),
        self._upper : (1,) + input_shape[:1] + (1, 1),
      }

class CenteredWeightAffine(builder.Affine):
  _count = 0
  def __init__(self, hidden_number, no_bias=False, initializer=None):
    super(CenteredWeightAffine, self).__init__(hidden_number, no_bias, initializer)
    label = 'certered-weight-affine%d' % CenteredWeightAffine._count
    self._weight = label + '_weight'
    self._bias = label + '_bias'
    CenteredWeightAffine._count += 1

  def forward(self, inputs, params):
    weight = params[self._weight]
    bias = params[self._bias]
#   weight_mean = np.mean(weight, axis=0).reshape((1, weight.shape[0]))
    weight_mean = np.sum(weight, axis=0) / float(weight.shape[0])
    weight = weight - weight_mean
    X_dot_W = np.dot(inputs, weight)
    if self._no_bias:
      output = X_dot_W
    else:
      output = X_dot_W + bias
    return output

class WNAffine(builder.Affine):
  _count = 0
  def __init__(self, hidden_number, no_bias=False, initializer=None):
    super(WNAffine, self).__init__(hidden_number, no_bias, initializer)
    label = 'wn-affine%d' % WNAffine._count
    self._weight = label + '_weight'
    self._bias = label + '_bias'
    self._gain = label + '_gain'
    WNAffine._count += 1

  def forward(self, inputs, params):
    weight = params[self._weight]
    bias = params[self._bias]
    gain = params[self._gain]

    weight_norm = np.sqrt(np.sum(weight ** 2, axis=0))
    weight_norm = weight_norm.reshape((1, weight.shape[1]))
    weight /= weight_norm
    weight *= gain

    outputs = np.dot(inputs, weight)
    if not self._no_bias:
      outputs += bias
    return outputs

  def parameter_shape(self, input_shape):
    shapes = super(WNAffine, self).parameter_shape(input_shape) 
    shapes[self._gain] = shapes[self._bias]
    return shapes

  def parameter_settings(self):
    settings = super(WNAffine, self).parameter_settings() 
    settings[self._gain] = {
      'init_rule'   : 'constant',
      'init_config' : { 'value' : 1.0 },
    }
    return settings

# blob normalization
def blob_normalization(
  X,
  settings,
  gamma,
  beta,
  mode='train',
  epsilon=1e-5,
  momentum=0.9,
  running_mean=None,
  running_variance=None
):
  N, D = map(int, X.shape)
  size = N * D

  if running_mean is None:
    running_mean = np.zeros(1)
  if running_variance is None:
    running_variance = np.zeros(1)

  if mode == 'train':
    if 'shared_mean' in settings:
      mean = np.sum(X) / size
    else:
      mean = np.sum(X, axis=0) / N
      mean = np.reshape(mean, (1, D))

    centered_X = X - mean

    if 'shared_deviation' in settings:
      variance = np.sum(centered_X ** 2) / size
    else:
      variance = np.sum(centered_X ** 2, axis=0) / N
      variance = np.reshape(variance, (1, D))

    deviation = variance ** 0.5
    rescaled_X = centered_X / deviation

    out = gamma * rescaled_X + beta

    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_variance = momentum * running_variance + (1.0 - momentum) * variance

  elif mode == 'test':
    X_hat = (X - running_mean) / np.sqrt(running_variance + epsilon)
    out = gamma * X_hat + beta

  return out, running_mean, running_variance

class BlobNormalization(builder.Module):
  count = 0
  def __init__(self, settings='', epsilon=1e-5, momentum=0.9):
    super(BlobNormalization, self).__init__()
    self.settings = settings
    self.epsilon  = epsilon
    self.momentum = momentum
    self.running_mean, self.running_variance = None, None

    self.gamma = 'bn%d_gamma' % self.__class__.count
    self.beta  = 'bn%d_beta' % self.__class__.count

    self.__class__.count += 1

  def forward(self, inputs, params):
    outputs, self.running_mean, self.running_variance = blob_normalization(
      inputs,
      self.settings,
      params[self.gamma],
      params[self.beta],
      params['__training_in_progress__'],
      self.epsilon,
      self.momentum,
      self.running_mean,
      self.running_variance
    )
    return outputs

  def output_shape(self, input_shape):
    return input_shape

  def parameter_shape(self, input_shape):
    return {
      self.gamma : (1,) if 'shared_mean' in self.settings else input_shape,
      self.beta  : (1,) if 'shared_deviation' in self.settings else input_shape,
    }

  def parameter_settings(self):
    return {
        self.gamma : {'init_rule' : 'constant', 'init_config' : {'value': 1.0}},
        self.beta  : {'init_rule' : 'constant', 'init_config' : {}}
    }

class ChannelMultiplication(builder.Module):
  _count = 0
  def __init__(self, co=None):
    if co is None:
      self._coefficient = 'channel_multiplication_%d' % ChannelMultiplication._count
    else:
      self._coefficient = co
    ChannelMultiplication._count += 1
  def forward(self, inputs, p):
    if isinstance(self._coefficient, str):
      coefficient = p[self._coefficient]
    else:
      coefficient = self._coefficient
    return coefficient * inputs
  def output_shape(self, input_shape):
    return input_shape
  def parameter_shape(self, input_shape):
    if isinstance(self._coefficient, str):
      return {self._coefficient : input_shape}
    else:
      return {}
  def parameter_settings(self):
    if isinstance(self._coefficient, str):
      return {self._coefficient : {'init_rule' : 'constant', 'init_config' : {'value' : 1.0}}}
    else:
      return {}

class ChannelDivision(builder.Module):
  _count = 0
  def __init__(self, co=None):
    if co is None:
      self._coefficient = 'channel_multiplication_%d' % ChannelDivision._count
    else:
      self._coefficient = co
    ChannelDivision._count += 1
  def forward(self, inputs, p):
    if isinstance(self._coefficient, str):
      coefficient = p[self._coefficient]
    else:
      coefficient = self._coefficient
    return inputs / coefficient
  def output_shape(self, input_shape):
    return input_shape
  def parameter_shape(self, input_shape):
    if isinstance(self._coefficient, str):
      return {self._coefficient : input_shape}
    else:
      return {}
  def parameter_settings(self):
    if isinstance(self._coefficient, str):
      return {self._coefficient : {'init_rule' : 'constant', 'init_config' : {'value' : 1.0}}}
    else:
      return {}

def layer_normalization(X, gamma, beta, epsilon=0.001):
  N, D = X.shape
  mean = np.sum(X, axis=1).reshape((N, 1)) / float(D)
  X = X - mean
  variance = np.sum(X ** 2, axis=1).reshape((N, 1)) /float(D)
  std = variance ** 0.5
  X = X / (std + epsilon)
  return X * gamma + beta

if __name__ == '__main__':
  X = np.random.normal(3, 3, (16, 2048))
  Y = layer_normalization(X, 1, 0)
  print np0.mean(to_np(Y), axis=1)
  print np0.std(to_np(Y), axis=1)

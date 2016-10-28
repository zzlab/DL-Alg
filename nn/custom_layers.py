import minpy.numpy as np
import minpy.nn.model_builder as builder

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

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

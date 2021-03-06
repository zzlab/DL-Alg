import mxnet as mx

class HybridInitializer(mx.initializer.Initializer):
  def __init__(self, parameters, initializer):
    self._parameters = parameters
    self._initializer = initializer

  def __call__(self, id, array):
    if id in self._parameters:
      array[:] = self._parameters[id]
    else: self._initializer(id, array)

class PReLUInitializer(mx.initializer.Xavier):
  def __init__(self):
    super(PReLUInitializer, self).__init__('gaussian', 'in', 2.0)

class DReLUInitializer(mx.initializer.Xavier):
  def __init__(self, magnitude=3.0, dictionary={}):
    super(DReLUInitializer, self).__init__('gaussian', 'in', magnitude=magnitude)
    self.dictionary = dictionary
  def __call__(self, name, array):
    print name, array.shape, self.dictionary[name].shape
    if name in self.dictionary:
      array[:] = self.dictionary[name]
    else:
      super(DReLUInitializer, self).__call__(name, array)
  def _init_default(self, arg, array):
    if 'lower' in arg:
      array[:] = 0.0
    elif 'upper' in arg:
      array[:] = 1.0

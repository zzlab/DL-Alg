import mxnet as mx

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

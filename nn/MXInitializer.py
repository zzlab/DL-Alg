import mxnet as mx

class DReLUInitializer(mx.initializer.Xavier):
  def __init__(self, magnitude=5.0, dictionary={}):
    super(DReLUInitializer, self).__init__('gaussian', 'in', magnitude=magnitude)
    self.dictionary = dictionary
  def __call__(self, name, array):
    if name in self.dictionary:
      print name, 'initialized'
      array[:] = self.dictionary[name]
    else:
      super(DReLUInitializer, self).__call__(name, array)
  def _init_default(self, arg, array):
    if 'lower' in arg:
      array[:] = 0.0
    elif 'upper' in arg:
      array[:] = 1.0

import mxnet as mx

class DReLUInitializer(mx.initializer.Xavier):
  def __init__(self, magnitude=1.0):
    super(DReLUInitializer, self).__init__(magnitude=magnitude)
  def _init_default(self, arg, array):
    if 'lower' in arg:
      array[:] = 0.0
    elif 'upper' in arg:
      array[:] = 2.0

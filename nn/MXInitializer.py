import mxnet as mx

class DReLUInitializer(mx.initializer.Xavier):
  def __init__(self):
    super(DReLUInitializer, self).__init__(magnitude=2.0)
  def _init_default(self, arg, array):
    if 'lower' in arg:
      array[:] = 0.0
    elif 'upper' in arg:
      array[:] = 1.0

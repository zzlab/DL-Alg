class MannualScheduler:
  def __init__(self, lr):
    self.lr = lr
  def __call__(self, *args):
    return self.lr

class FactorScheduler:
  def __init__(self, lr, factor, interval):
    self._lr = lr
    self._factor = factor
    self._interval = interval
  def __call__(self, iteration):
    if (iteration + 1) % self._interval:
      self._lr *= self._factor
    return self._lr

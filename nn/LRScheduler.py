class MannualScheduler:
  def __init__(self, lr):
    self.lr = lr
  def __call__(self, *args):
    return self.lr

class FactorScheduler:
  def __init__(self, lr, factor, interval):
    self.lr = lr
    self._factor = factor
    self._interval = interval
    self._previous_iteration = None
  def __call__(self, iteration):
    if iteration != self._previous_iteration:
      if iteration != 0 and iteration % self._interval == 0:
        self.lr *= self._factor
      self._previous_iteration = iteration
    return self.lr

class DecayingAtEpochScheduler:
  def __init__(self, lr, factor, epochs_to_decay, batches):
    # epoch starts from 0
    self.lr = lr
    self._factor = factor
    self._epochs_to_decay = epochs_to_decay + [float('inf')]
    self._epoch_index = 0
    self._batches = batches
    self._previous_iteration = None
  def __call__(self, iteration):
    if iteration != self._previous_iteration:
      if iteration / self._batches == self._epochs_to_decay[self._epoch_index]:
        self.lr *= self._factor
        print 'decay at', iteration
        self._epoch_index += 1
      self._previous_iteration = iteration
    return self.lr

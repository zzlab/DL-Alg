class MannualScheduler:
  def __init__(self, lr):
    self.lr = lr
  def __call__(self, *args):
    return self.lr

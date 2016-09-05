class BoundMonitor:
  def __init__(self):
    self.record = []
  def __call__(self, epoch, symbol, args, states):
    self.record.append({})
    for key, value in args.items():
      if 'lower' in key or 'upper' in key:
        self.record[-1].update({key : value.asnumpy()})

import cPickle as pickle

class EpochCheckpoint(object):
  def __init__(self, prefix, path):
    self.prefix = prefix
    self.path = path
  def __call__(self, epoch, symbol, args, states):
    with open('%s/%s-epoch-%d-parameter' % (self.path, self.prefix, epoch), 'wb') as output:
      pickle.dump((args, states), output)

class BoundsCheckpoint(EpochCheckpoint):
  def __init__(self, prefix, path):
    super(BoundsCheckpoint, self).__init__(prefix, path)
    self.record = []
  def __call__(self, *args):
    raise NotImplementedError()
  def batch_end_callback(self, args):
    bounds = {}
    for key, value in args.locals['arg_params'].items():
      if 'lower' in key or 'upper' in key:
        bounds[key] = value
    self.record.append(bounds)
  def epoch_end_callback(self, epoch, symbol, args, states):
    super(BoundsCheckpoint, self).__call__(epoch, symbol, args, states)
    with open('%s/%s-epoch-%d-bounds' % (self.path, self.prefix, epoch), 'wb') as output:
      pickle.dump(self.record, output)
    del self.record[:]

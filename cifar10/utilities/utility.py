import mxnet as mx

import sys
sys.path.append('../..')

def forward_layer(inputs, args):
  if isinstance(args[1], dict):
    return getattr(__import__('layers'), args[0])(inputs, **args[1])
  else:  
    return getattr(__import__('layers'), args[0])(inputs, *args[1:])

def accuracy(p,l):
  import minpy.numpy as np
  if len(l.shape) == 1:
    return 1-np.count_nonzero(p-l).val/float(p.shape[0])
  else:
    inputs, labels = p, l
    return np.mean(np.sum((inputs-labels)**2, axis=1))

def sparsity(l):
  return 1-np.count_nonzero(l).val/float(l.asnumpy().size)

def averaged_top_n(l, n):
  return sum(sorted(l, reverse=True)[:n]) / float(n)

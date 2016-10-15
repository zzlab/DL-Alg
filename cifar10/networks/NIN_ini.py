import cPickle as pickle
import minpy.numpy as np
from minpy.nn.io import NDArrayIter
import minpy.nn.model_builder as builder
from minpy.nn.solver import Solver

import sys
sys.path.append('../')
sys.path.append('../../nn')

from custom_layers import DReLU

from utilities.data_utility import load_cifar10

activation = sys.argv[1]
# activation = 'ReLU'
if activation == 'ReLU':
  network_in_network = builder.Sequential(
    builder.Convolution((5, 5), 192, pad=(2, 2)),
    builder.ReLU(),
    builder.Convolution((1, 1), 160),
    builder.ReLU(),
    builder.Convolution((1, 1), 96),
    builder.ReLU(),
    builder.Pooling('max', (3, 3), (2, 2), (1, 1)),
    builder.Dropout(0.5),
    builder.Convolution((5, 5), 192, pad=(2, 2)),
    builder.ReLU(),
    builder.Convolution((1, 1), 192),
    builder.ReLU(),
    builder.Convolution((1, 1), 192),
    builder.ReLU(),
    builder.Pooling('avg', (3, 3), (2, 2), (1, 1)),
    builder.Dropout(0.5),
    builder.Convolution((3, 3), 192, pad=(1, 1)),
    builder.ReLU(),
    builder.Convolution((1, 1), 192),
    builder.ReLU(),
    builder.Convolution((1, 1), 10),
    builder.ReLU(),
    builder.Pooling('avg', (8, 8)),
    builder.Reshape((10,))
  )
else:
  network_in_network = builder.Sequential(
    builder.Convolution((5, 5), 192, pad=(2, 2)),
    DReLU(),
    builder.Convolution((1, 1), 160),
    DReLU(),
    builder.Convolution((1, 1), 96),
    DReLU(),
    builder.Pooling('max', (3, 3), (2, 2), (1, 1)),
    builder.Dropout(0.5),
    builder.Convolution((5, 5), 192, pad=(2, 2)),
    DReLU(),
    builder.Convolution((1, 1), 192),
    DReLU(),
    builder.Convolution((1, 1), 192),
    DReLU(),
    builder.Pooling('avg', (3, 3), (2, 2), (1, 1)),
    builder.Dropout(0.5),
    builder.Convolution((3, 3), 192, pad=(1, 1)),
    DReLU(),
    builder.Convolution((1, 1), 192),
    DReLU(),
    builder.Convolution((1, 1), 10),
    DReLU(),
    builder.Pooling('avg', (8, 8)),
    builder.Reshape((10,))
  )


data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)

ini_mode = sys.argv[2]
# ini_mode = 'normal'
if ini_mode == 'layer-by-layer':
  model = builder.Model(network_in_network, 'softmax', (3, 32, 32,), data[2])
  solver = Solver(
    model,
    NDArrayIter(data[0], data[1]),
    NDArrayIter(data[0], data[1]),
  )
  solver.init()
else:
  model = builder.Model(network_in_network, 'softmax', (3, 32, 32,))
  for arg, setting in model.param_configs.items():
    print arg
    shape = setting['shape']
    if 'weight' in arg:
      if len(shape) == 2:
        n = shape[0]
      elif len(shape) == 4:
        n = np.prod(shape[1:])
      else:
        raise Exception()
      std = (2 / float(n)) ** 0.5
      model.params[arg] = np.random.normal(0, std, shape)
    elif 'bias' in arg:
      model.params[arg] = np.zeros(shape)
    elif 'lower' in arg:
      model.params[arg] = np.zeros(shape)
    elif 'upper' in arg:
      model.params[arg] = np.ones(shape)
    else:
      raise Exception()

parameters = {key : value.asnumpy() for key, value in model.params.items()}
output_file = 'NIN-%s-%s-initial-parameters' % (activation, ini_mode)
pickle.dump(parameters, open(output_file, 'wb'))

import cPickle as pickle
from minpy.nn.io import NDArrayIter
import minpy.nn.model_builder as builder
from minpy.nn.solver import Solver

import sys
sys.path.append('../')
sys.path.append('../../nn')

from utilities.data_utility import load_cifar10

network_in_network = builder.Sequential(
  builder.Reshape((3, 32, 32)),
  builder.Convolution((5, 5), 192, pad=(2, 2)),
  builder.ReLU(),
  builder.Convolution((1, 1), 160),
  builder.ReLU(),
  builder.Convolution((1, 1), 96),
  builder.ReLU(),
  builder.Pooling('max', (3, 3), (2, 2)),
  builder.Dropout(0.5),
  builder.Convolution((5, 5), 192, pad=(2, 2)),
  builder.ReLU(),
  builder.Convolution((1, 1), 192),
  builder.ReLU(),
  builder.Convolution((1, 1), 192),
  builder.ReLU(),
  builder.Pooling('avg', (3, 3), (2, 2)),
  builder.Dropout(0.5),
  builder.Convolution((3, 3), 192, pad=(2, 2)),
  builder.ReLU(),
  builder.Convolution((1, 1), 192),
  builder.ReLU(),
  builder.Convolution((1, 1), 10),
  builder.ReLU(),
  builder.Pooling('avg', (8, 8)),
  builder.Reshape((10,))
)

data = load_cifar10(path='../utilities/cifar/', reshape=True, center=True, rescale=True)
model = builder.Model(network_in_network, 'softmax', (3, 32, 32,), data[0])
solver = Solver(
  model,
  NDArrayIter(data[0], data[1]),
  NDArrayIter(data[0], data[1]),
)
solver.init()
parameters = {key : value.asnumpy() for key, value in model.params.items()}
pickle.dump(parameters, open('NIN_ReLU_ini', 'wb'))

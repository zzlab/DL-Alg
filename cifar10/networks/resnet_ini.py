import cPickle as pickle
import marshal
import mxnet as mx
import numpy as np

import sys

sys.path.append('..')
from utilities.data_utility import load_cifar10
data = load_cifar10('../utilities/cifar/', reshape=True)

sys.path.append('../../nn')

from MXInitializer import DReLUInitializer
initializer = DReLUInitializer(1.0)

from MXModels.ResidualNetwork import ResidualNetwork

# activation = 'ReLU'
activation = sys.argv[1]
symbol, _ = ResidualNetwork(3, activation, initializer)(data[0].shape)

args = symbol.list_arguments()
shapes, _, _ = symbol.infer_shape(data=data[0].shape)

parameters = {}
for arg, shape in zip(args, shapes):
  print arg
  parameters[arg] = mx.nd.zeros(shape, mx.cpu())
  initializer(arg, parameters[arg])

for key, value in parameters.items():
  parameters[key] = value.asnumpy()
  parameters[key] = parameters[key].astype(np.float16)

output_file = 'resnet-%s-initial-parameters' % (activation)
pickle.dump(parameters, open(output_file, 'wb'))

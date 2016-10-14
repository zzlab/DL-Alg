import minpy
from minpy.core import grad_and_loss as gradient_loss
from minpy.nn.io import NDArrayIter
from minpy.nn import model_builder as builder
from minpy.nn.solver import Solver

from model_gallery import MultiLayerPerceptron as MLP

import sys
sys.path.append('../../')
sys.path.append('../../../nn')
from custom_layers
from utilities.data_utility import load_cifar10

training_X, training_Y, _, _, test_X, test_Y = \
  load_cifar10(path='../../utilities/cifar/', center=True, rescale=True)

HIDDEN_LAYERS = 4
activation = sys.argv[1]
# activation = 'ReLU'
storage = {}
mlp = MLP(
  *((1024,) * HIDDEN_LAYERS + (10,)),
  activation         = activation,
  affine_monitor     = True,
  activation_monitor = False,
  storage            = storage
)

ini_mode = sys.argv[2]
# ini_mode = 'layer-by-layer'
if ini_mode == 'layer-by-layer':
  model = builder.Model(mlp, 'softmax', (3072,), training_X)
else:
  model = builder.Model(mlp, 'softmax', (3072,))

solver = Solver(
  model,
  NDArrayIter(training_X, training_Y),
  NDArrayIter(test_X, test_Y),
  init_rule = 'xavier'
)
 
solver.init()

model.forward(test_X, 'train')

for key, value in storage.items():
  storage[key] = value.asnumpy()

import cPickle as pickle
pickle.dump(storage, open('pre-activation-value-%s-%s' % (activation, ini_mode), 'wb'))

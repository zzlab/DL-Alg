import minpy
from minpy.core import grad_and_loss as gradient_loss
from minpy.nn.io import NDArrayIter
from minpy.nn import model_builder as builder
from minpy.nn.solver import Solver

from model_gallery import MultiLayerPerceptron as MLP

import sys
sys.path.append('../../')
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
  affine_monitor     = False,
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

parameter_keys = list(model.params.keys())
parameter_values = list(model.params.values())
def loss_function(*args):
  predictions = model.forward(test_X, 'train')
  return model.loss(predictions, test_Y)
gl = gradient_loss(loss_function, range(len(parameter_keys)))
gradients, loss = gl(*parameter_values)
mapped_gradients = dict(zip(parameter_keys, gradients))
for key, value in mapped_gradients.items():
  mapped_gradients[key] = value.asnumpy()

import cPickle as pickle
pickle.dump(mapped_gradients, open('initial-gradient-%s-%s' % (activation, ini_mode), 'wb'))

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
# activation = sys.argv[1]
activation = 'ReLU'
storage = {}
mlp = MLP(
  *((1024,) * HIDDEN_LAYERS + (10,)),
  activation         = activation,
  affine_monitor     = False,
  activation_monitor = False,
  storage            = storage
)

# ini_mode = sys.argv[2]
ini_mode = 'layer-by-layer'
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

parameter_keys = model.params.keys()
parameter_values = model.params.values()
predictions = model.forward(test_X, 'train')
def loss_function(*args):
  return model.loss(predictions, test_Y)

checkpoint_loss = (2.0, 1.5, 1.0, 0.5)

lr = 0.01
for loss_value in checkpoint_loss:
  while True:
    gl = gradient_loss(loss_function, range(len(parameter_keys)))
    gradients, loss = gl(*parameter_values)
    print type(loss), loss
    mapped_gradients = dict(zip(parameter_keys, gradients))
    for key, value in mapped_gradients.items():
      model.params[key] -= lr * value

    '''
    for key, value in mapped_gradients.items():
      mapped_gradients[key] = value.asnumpy()
    '''

import cPickle as pickle
pickle.dump(mapped_gradients, open('initial-gradient-%s-%s' % (activation, ini_mode), 'wb'))

import cPickle as pickle

import minpy
import minpy.numpy as np

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

checkpoint_loss = (2.0, 1.5, 1.0, 0.5)

lr = 0.05
batch_size = 100
batch_count = len(training_X) // batch_size
batch_index = 0

def loss_function(X, Y, *args):
  predictions = model.forward(X, 'train')
  return model.loss(predictions, Y)
gl = gradient_loss(loss_function, range(2, len(model.params) + 2))

for loss_value in checkpoint_loss:
  while True:
    # batch
    print batch_index
    data_batch = training_X[batch_index * batch_size : (batch_index + 1) * batch_size]
    label_batch = training_Y[batch_index * batch_size : (batch_index + 1) * batch_size]

    # update batch index
    batch_index = (batch_index + 1) % (batch_count - 1)

    # compute gradients and loss
    parameter_keys = list(model.params.keys())
    parameter_values = list(model.params.values())
    gradients, loss = gl(test_X, test_Y, *parameter_values)
    print loss
    loss = round(loss[0], 1)

    # update networks
    mapped_gradients = dict(zip(parameter_keys, gradients))
    for key, value in mapped_gradients.items():
      model.params[key] -= lr * value

    if loss == loss_value:
      # checkpoint
      for key, value in mapped_gradients.items():
        mapped_gradients[key] = value.asnumpy()
      pickle.dump(mapped_gradients, open('training-gradient-loss-%f' % loss_value, 'wb'))
      break

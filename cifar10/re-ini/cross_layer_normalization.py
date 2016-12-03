import numpy as np0

import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

def cl_rescale(container, divisions, X, parameters, epsilon=1E-3, activations=(ReLU, Sigmoid, Tanh)):
  activation_count = 0
  for index, module in enumerate(container):
    X = module.forward(X, parameters)
    if isinstance(module, activations):
      std = array_std(X, axis=0)
      index = np0.abs(std) < epsilon
      std[index] = 1
      divisions[activation_count]._coefficient[:] = std
      next_weight = 'fully_connected%d_weight' % (activation_count + 1)
      if next_weight in parameters:
        parameters[next_weight] *= std.reshape((len(std), 1))
      activation_count += 1

'''
ACTIVATION = sys.argv[1]
DEVICE = int(sys.argv[2])
DR_INTERVAL = int(sys.argv[3])
shapes = [int(shape) for shape in sys.argv[4:]]
'''

ACTIVATION = 'ReLU'
DEVICE = 0
DR_INTERVAL = 10
shapes = (1024,) * 4 + (10,)

activation = getattr(builder, ACTIVATION)
set_context(gpu(DEVICE))

storage = {}
chd_list = []
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(builder.Export('affine%d' % i, storage))
  mlp.append(activation())
  mlp.append(builder.Export('activation%d' % i, storage))
  mlp.append(ChannelDivision(np.ones(shape)))
  chd_list.append(mlp[-1])
  mlp.append(builder.Export('chd%d' % i, storage))

mlp.append(builder.Affine(shapes[-1]))
model = builder.Model(mlp, 'softmax', (3072,))

batch_size = 100
batches = len(data[0]) // batch_size
batch_index = 0

iterations = 25000
interval = 10

settings = {'learning_rate' : 0.05}
initialize(model)
updater = Updater(model, 'sgd', settings)

for key, value in model.params.items():
  print key, value.context

loss_history = []
for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = to_float(loss)
  loss_history.append(loss)

  updater.update(gradients)

  if (i + 1) % DR_INTERVAL == 0:
    cl_rescale(mlp, chd_list, X_batch, model.params)
    model.forward(X_batch, 'train')

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

configuration = '%s-interval-%s-shape-%s' % (ACTIVATION, DR_INTERVAL, '-'.join(str(d) for d in shapes))
pickle.dump(loss_history, open('model/mlp-cln-loss-%s' % configuration, 'wb'))

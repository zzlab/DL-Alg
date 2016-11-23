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

# ACTIVATION = sys.argv[1]
ACTIVATION = 'Sigmoid'
activation = getattr(builder, ACTIVATION)
# DEVICE = int(sys.argv[2])
DEVICE = 1
set_context(gpu(DEVICE))
# DR_INTERVAL = int(sys.argv[3])
DR_INTERVAL = 10
# shapes = [int(shape) for shape in sys.argv[4:]]
shapes = (1024,) * 4 + (10,)

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

def cl_rescale(container, divisions, X, parameters, epsilon=1E-3, activations=(ReLU, Sigmoid, Tanh)):
  '''
  for chd in divisions:
    print 'coefficient', chd._coefficient
  '''
  activation_count = 0
  for index, module in enumerate(container):
    X = module.forward(X, parameters)
#   print type(module), np_min(X), np_max(X)
    if isinstance(module, activations):
#     std = np0.maximum(array_std(X, axis=0), epsilon)
      std = array_std(X, axis=0)
      index = np0.abs(std) < epsilon
      std[index] = 1
      print np0.min(std), np0.max(std)
      divisions[activation_count]._coefficient[:] = std
      next_weight = 'fully_connected%d_weight' % (activation_count + 1)
      if next_weight in parameters:
        parameters[next_weight] *= std.reshape((len(std), 1))
      activation_count += 1
# raise Exception()

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
    '''
    cl_rescale(mlp, chd_list, X_batch, model.params)
    model.forward(X_batch, 'train')
    '''
    '''
    for key, value in storage.items():
      if 'activation' in key:
        std = array_std(value, axis=0)
        print key, np_min(std), np_max(std)
    '''
    '''
    for key, value in model.params.items():
      if 'weight' in key:
        std = array_std(value, axis=0)
        print key, np_min(std), np_max(std)
    '''
    '''
    for key, value in storage.items():
      if 'chd' in key:
        std = array_std(value, axis=0)
        print key, np_min(std), np_max(std)
    '''

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

  '''
  print 'iteration %d loss %f' % (i + 1, loss)
  '''
  '''
  for key, value in storage.items():
    if 'activation' in key:
      value = to_np(value)
      print key, n_channel
  '''
  '''
  if i == 50:
    raise Exception()
  '''

configuration = '%s-interval-%s-shape-%s' % (ACTIVATION, DR_INTERVAL, '-'.join(str(d) for d in shapes))
pickle.dump(loss_history, open('model/mlp-cln-loss-%s' % configuration, 'wb'))

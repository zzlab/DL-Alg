import numpy as np0

import minpy.numpy as np
import minpy.nn.model_builder as builder

from minpy.context import set_context, cpu, gpu
DEVICE = 2
set_context(gpu(DEVICE))

import cPickle as pickle

import sys
sys.path.append('../../nn')
from custom_layers import *
from facility import *
from solver_primitives import *

sys.path.append('../')
from utilities.data_utility import load_cifar10
data = load_cifar10(path='../utilities/cifar/', center=True, rescale=True)

hidden_layers = 4
shapes = (1024,) * hidden_layers + (10,)
ACTIVATION = 'Sigmoid'
activation = getattr(builder, ACTIVATION)
storage = {}
mlp = builder.Sequential()
for i, shape in enumerate(shapes[:-1]):
  mlp.append(builder.Affine(shape))
  mlp.append(builder.Export('affine%d' % i, storage))
  mlp.append(activation())
  mlp.append(builder.Export('activation%d' % i, storage))
mlp.append(builder.Affine(shapes[-1]))
mlp.append(builder.Export('affine%d' % (len(shapes) - 1), storage))
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
mean = {key : [] for key in model.params}
std = {key : [] for key in model.params}
minimum = {key : [] for key in model.params}
maximum = {key : [] for key in model.params}

channel_sparsity = {key : [] for key in storage if 'activation' in key}
print channel_sparsity

for i in range(iterations):
  X_batch = data[0][batch_index * batch_size : (batch_index + 1) * batch_size]
  Y_batch = data[1][batch_index * batch_size : (batch_index + 1) * batch_size]
  batch_index = (batch_index + 1) % batches

  gradients, loss = gradient_loss(model, X_batch, Y_batch)
  loss = to_float(loss)
  loss_history.append(loss)

  '''
  for key, value in storage.items():
    if 'activation' in key:
      value = to_np(value)
      n_channel = sum(np0.all(value[:, i] == 0) for i in range(value.shape[1]))
      ratio = n_channel / 1024.0
      channel_sparsity[key].append(n_channel)
  '''
#     print key, ratio

  '''
  for key, value in zip(model.params.keys(), gradients):
    mean[key].append(np.mean(value).asnumpy())
    std[key].append(np.std(value).asnumpy())
    minimum[key].append(np.min(value).asnumpy())
    maximum[key].append(np.max(value).asnumpy())
  '''

  updater.update(gradients)

  if (i + 1) % interval == 0:
    print 'iteration %d loss %f' % (i + 1, loss)

# pickle.dump((loss_history, mean, std, minimum, maximum), open('forward-statistics', 'wb'))
# pickle.dump((loss_history, channel_sparsity), open('channel-sparsity-%s' % ACTIVATION, 'wb'))
pickle.dump(loss_history, open('model/loss-%s' % ACTIVATION, 'wb'))
